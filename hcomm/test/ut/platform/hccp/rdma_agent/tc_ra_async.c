/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <errno.h>
#include <stdlib.h>
#include <pthread.h>
#include "securec.h"
#include "ut_dispatch.h"
#include "dl_hal_function.h"
#include "ra_rs_comm.h"
#include "hccp_common.h"
#include "ra_async.h"
#include "ra.h"
#include "ra_hdc.h"
#include "ra_hdc_ctx.h"
#include "ra_hdc_async.h"
#include "ra_hdc_async_ctx.h"
#include "ra_hdc_async_socket.h"
#include "ra_client_host.h"
#include "ra_comm.h"
#include "tc_ra_async.h"

extern int RaHdcAsyncInitSession(struct RaInitConfig* cfg);
extern int RaHdcAsyncMutexInit(unsigned int phyId);
extern void RaHwAsyncSetConnectStatus(unsigned int phyId, unsigned int connectStatus);
extern int RaHdcAsyncSessionClose(unsigned int phyId);
extern struct HdcAsyncInfo gRaHdcAsync[RA_MAX_PHY_ID_NUM];
extern int RaSocketSendAsync(
    const void* fdHandle, const void* data, unsigned long long size, unsigned long long* sentSize, void** reqHandle);
extern int RaSocketRecvAsync(
    const void* fdHandle, void* data, unsigned long long size, unsigned long long* receivedSize, void** reqHandle);
extern int RaGetAsyncReqResult(void* reqHandle, int* reqResult);
extern int RaSocketBatchConnectAsync(struct SocketConnectInfoT conn[], unsigned int num, void** reqHandle);
extern int RaSocketListenStartAsync(struct SocketListenInfoT conn[], unsigned int num, void** reqHandle);
extern int RaSocketListenStopAsync(struct SocketListenInfoT conn[], unsigned int num, void** reqHandle);
extern int RaSocketBatchCloseAsync(struct SocketCloseInfoT conn[], unsigned int num, void** reqHandle);

int RaHdcSendMsgAsyncStub(
    unsigned int opcode, unsigned int phyId, char* data, unsigned int dataSize, struct RaRequestHandle* reqHandle)
{
    reqHandle->isDone = true;
    return 0;
}

void HdcAsyncDelResponseStub(struct RaRequestHandle* reqHandle)
{
    free(reqHandle);
    reqHandle = NULL;
    return;
}

void* CallocFirstStub(unsigned long num, unsigned long size)
{
    static int callocFirstStubCallNum1 = 0;
    callocFirstStubCallNum1++;

    if (callocFirstStubCallNum1 == 1) {
        return calloc(num, size);
    }

    return NULL;
}

void TcRaCtxLmemRegisterAsync()
{
    struct RaCtxHandle ctxHandle = {0};
    struct MrRegInfoT lmemInfo = {0};
    struct RaLmemHandle* lmemHandle = NULL;
    struct RaRequestHandle* reqHandle = NULL;
    union OpLmemRegInfoData asyncData = {0};
    struct MemRegInfo info = {0};
    int ret;

    mocker_clean();
    ctxHandle.protocol = 1;

    mocker(RaHdcCtxLmemRegisterAsync, 10, -1);
    ret = RaCtxLmemRegisterAsync(&ctxHandle, &lmemInfo, (void**)&lmemHandle, (void**)&reqHandle);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker(RaHdcSendMsgAsync, 10, 0);
    ret = RaCtxLmemRegisterAsync(&ctxHandle, &lmemInfo, (void**)&lmemHandle, (void**)&reqHandle);
    EXPECT_INT_EQ(ret, 0);

    reqHandle->recvBuf = &asyncData;
    reqHandle->privData = &info;
    mocker(memcpy_s, 10, -1);
    RaHdcAsyncHandleLmemRegister(reqHandle);
    free(lmemHandle);
    mocker_clean();
    free(reqHandle);
}

void TcRaCtxLmemUnregisterAsync()
{
    struct RaCtxHandle ctxHandle = {0};
    struct RaLmemHandle* lmemHandle = malloc(sizeof(struct RaLmemHandle));
    void* reqHandle = NULL;

    mocker_clean();

    mocker(RaHdcSendMsgAsync, 1, 0);
    RaCtxLmemUnregisterAsync(&ctxHandle, lmemHandle, &reqHandle);
    mocker_clean();
    free(reqHandle);

    mocker(RaHdcSendMsgAsync, 1, 0);
    lmemHandle = malloc(sizeof(struct RaLmemHandle));
    RaCtxLmemUnregisterAsync(&ctxHandle, lmemHandle, &reqHandle);
    mocker_clean();
    free(reqHandle);

    mocker(RaHdcCtxLmemUnregisterAsync, 1, -1);
    lmemHandle = malloc(sizeof(struct RaLmemHandle));
    RaCtxLmemUnregisterAsync(&ctxHandle, lmemHandle, &reqHandle);
    mocker_clean();
}

void TcRaCtxQpCreateAsync()
{
    struct RaCtxHandle ctxHandle = {0};
    struct QpCreateAttr qpAttr = {0};
    struct QpInfoT* qpInfo = NULL;
    struct RaCtxQpHandle* qpHandle = NULL;
    struct RaRequestHandle* reqHandle = NULL;
    struct RaCqHandle scqHandle = {0};
    struct RaCqHandle rcqHandle = {0};

    qpAttr.scqHandle = (void*)&scqHandle;
    qpAttr.rcqHandle = (void*)&rcqHandle;
    ctxHandle.protocol = 1;
    mocker_clean();

    mocker(RaHdcSendMsgAsync, 1, 0);
    RaCtxQpCreateAsync(&ctxHandle, &qpAttr, (struct QpCreateInfo*)&qpInfo, (void**)&qpHandle, (void**)&reqHandle);
    free(qpHandle);
    free(reqHandle);
    mocker_clean();

    mocker(RaHdcSendMsgAsync, 1, 0);
    RaCtxQpCreateAsync(&ctxHandle, &qpAttr, (struct QpCreateInfo*)&qpInfo, (void**)&qpHandle, (void**)&reqHandle);
    mocker_clean();
    free(qpHandle);
    free(reqHandle);

    mocker(RaHdcCtxQpCreateAsync, 1, -1);
    RaCtxQpCreateAsync(&ctxHandle, &qpAttr, (struct QpCreateInfo*)&qpInfo, (void**)&qpHandle, (void**)&reqHandle);
    mocker_clean();
}

void TcRaCtxQpDestroyAsync()
{
    struct RaCtxQpHandle* qpHandle = malloc(sizeof(struct RaCtxQpHandle));
    void* reqHandle = NULL;

    mocker_clean();

    mocker(RaHdcSendMsgAsync, 1, 0);
    RaCtxQpDestroyAsync(qpHandle, &reqHandle);
    mocker_clean();

    free(reqHandle);

    qpHandle = malloc(sizeof(struct RaCtxQpHandle));
    mocker(RaHdcCtxQpDestroyAsync, 1, -1);
    RaCtxQpDestroyAsync(qpHandle, &reqHandle);
    mocker_clean();
}

void TcRaCtxQpImportAsync()
{
    struct RaCtxHandle ctxHandle = {0};
    struct QpImportInfoT info = {0};
    struct RaCtxRemQpHandle* remQpHandle = NULL;
    struct RaRequestHandle* reqHandle = NULL;

    ctxHandle.protocol = 1;
    mocker_clean();

    mocker(RaHdcSendMsgAsync, 1, 0);
    RaCtxQpImportAsync(&ctxHandle, &info, (void**)&remQpHandle, (void**)&reqHandle);
    free(remQpHandle);
    mocker_clean();

    mocker(RaHdcCtxQpImportAsync, 1, -1);
    RaCtxQpImportAsync(&ctxHandle, &info, (void**)&remQpHandle, (void**)&reqHandle);
    mocker_clean();
    free(reqHandle);
}

void TcRaCtxQpUnimportAsync()
{
    struct RaCtxRemQpHandle* remQpHandle = malloc(sizeof(struct RaCtxRemQpHandle));
    void* reqHandle = NULL;
    mocker_clean();

    mocker(RaHdcSendMsgAsync, 1, 0);
    RaCtxQpUnimportAsync(remQpHandle, &reqHandle);
    mocker_clean();

    free(reqHandle);

    remQpHandle = malloc(sizeof(struct RaCtxRemQpHandle));
    mocker(RaHdcCtxQpUnimportAsync, 1, -1);
    RaCtxQpUnimportAsync(remQpHandle, &reqHandle);
    mocker_clean();
}

void TcRaSocketSendAsync()
{
    struct SocketHdcInfo fdHandle = {0};
    struct RaRequestHandle* reqHandle = NULL;
    unsigned long long sentSize = 0;

    mocker_clean();
    mocker(RaHdcSendMsgAsync, 1, 0);
    RaSocketSendAsync(&fdHandle, "a", 1, &sentSize, (void**)&reqHandle);
    mocker_clean();

    free(reqHandle);
}

void TcRaSocketRecvAsync()
{
    struct SocketHdcInfo fdHandle = {0};
    struct RaRequestHandle* reqHandle = NULL;
    unsigned long long receivedSize = 0;
    char data = 0;

    mocker_clean();
    mocker(RaHdcSendMsgAsync, 1, 0);
    RaSocketRecvAsync(&fdHandle, &data, 1, &receivedSize, (void**)&reqHandle);
    free(reqHandle->privData);
    mocker_clean();

    free(reqHandle);
}

void TcRaGetAsyncReqResult()
{
    struct RaRequestHandle* reqHandle = malloc(sizeof(struct RaRequestHandle));
    struct RaAsyncOpHandle opHandle = {0};
    int reqResult = 0;

    reqHandle->isDone = 1;
    reqHandle->phyId = 0;
    reqHandle->recvLen = 1;
    reqHandle->opHandle = &opHandle;
    RA_INIT_LIST_HEAD(&reqHandle->list);
    reqHandle->recvBuf = malloc(1);
    reqHandle->privHandle = NULL;
    mocker_clean();

    mocker(pthread_mutex_lock, 1, 0);
    mocker(pthread_mutex_unlock, 1, 0);
    RaGetAsyncReqResult(reqHandle, &reqResult);
    mocker_clean();
}

void TcRaSocketBatchConnectAsyncNormal()
{
    struct RaRequestHandle* reqHandle = NULL;
    struct RaSocketHandle socketHandle = {0};
    struct SocketConnectInfoT conn[1] = {0};
    int ret = 0;

    conn[0].socketHandle = (void*)&socketHandle;

    mocker_clean();
    mocker(RaInetPton, 10, 0);

    mocker(RaHdcSendMsgAsync, 10, 0);
    mocker(RaGetSocketConnectInfo, 10, 0);

    ret = RaSocketBatchConnectAsync(conn, 1, (void**)&reqHandle);
    free(reqHandle);
    mocker_clean();
    return;
}

void TcRaSocketBatchConnectAsyncFail()
{
    struct RaRequestHandle* reqHandle = NULL;
    struct RaSocketHandle socketHandle = {0};
    struct SocketConnectInfoT conn[1] = {0};
    int ret = 0;

    conn[0].socketHandle = (void*)&socketHandle;

    mocker_clean();
    mocker(RaInetPton, 10, 0);

    mocker(RaHdcSendMsgAsync, 10, 0);
    mocker(RaGetSocketConnectInfo, 10, 0);

    ret = RaSocketBatchConnectAsync(NULL, 1, (void**)&reqHandle);
    EXPECT_INT_NE(ret, 0);

    ret = RaSocketBatchConnectAsync(conn, 0, (void**)&reqHandle);
    EXPECT_INT_NE(ret, 0);

    ret = RaSocketBatchConnectAsync(conn, 1, NULL);
    EXPECT_INT_NE(ret, 0);

    conn[0].socketHandle = NULL;
    ret = RaSocketBatchConnectAsync(conn, 1, (void**)&reqHandle);
    EXPECT_INT_NE(ret, 0);

    mocker_clean();
    return;
}

void TcRaSocketBatchConnectAsync()
{
    TcRaSocketBatchConnectAsyncNormal();
    TcRaSocketBatchConnectAsyncFail();
}

void TcRaSocketListenStartAsyncNormal()
{
    struct RaRequestHandle* reqHandle = NULL;
    struct RaSocketHandle socketHandle = {0};
    struct SocketListenInfoT conn[1] = {0};
    int ret = 0;

    conn[0].socketHandle = (void*)&socketHandle;

    mocker_clean();
    mocker(RaInetPton, 10, 0);
    mocker(RaHdcSendMsgAsync, 10, 0);
    mocker(RaGetSocketListenInfo, 10, 0);

    ret = RaSocketListenStartAsync(conn, 1, (void**)&reqHandle);
    free(reqHandle->privData);
    free(reqHandle);
    mocker_clean();
    return;
}

void TcRaSocketListenStartAsyncFail()
{
    struct RaRequestHandle* reqHandle = NULL;
    struct RaSocketHandle socketHandle = {0};
    struct SocketListenInfoT conn[1] = {0};
    int ret = 0;

    conn[0].socketHandle = (void*)&socketHandle;

    mocker_clean();
    mocker(RaInetPton, 10, 0);
    mocker(RaHdcSendMsgAsync, 10, 0);
    mocker(RaGetSocketListenInfo, 10, 0);

    ret = RaSocketListenStartAsync(NULL, 1, (void**)&reqHandle);
    EXPECT_INT_NE(ret, 0);

    ret = RaSocketListenStartAsync(conn, 0, (void**)&reqHandle);
    EXPECT_INT_NE(ret, 0);

    ret = RaSocketListenStartAsync(conn, 1, NULL);
    EXPECT_INT_NE(ret, 0);

    conn[0].socketHandle = NULL;
    ret = RaSocketListenStartAsync(conn, 1, (void**)&reqHandle);
    EXPECT_INT_NE(ret, 0);

    mocker_clean();
    return;
}

void TcRaSocketListenStartAsync()
{
    TcRaSocketListenStartAsyncNormal();
    TcRaSocketListenStartAsyncFail();
}

void TcRaSocketListenStopAsyncNormal()
{
    struct RaRequestHandle* reqHandle = NULL;
    struct RaSocketHandle socketHandle = {0};
    struct SocketListenInfoT conn[1] = {0};

    conn[0].socketHandle = (void*)&socketHandle;

    mocker_clean();
    mocker(RaInetPton, 10, 0);
    mocker(RaHdcSendMsgAsync, 10, 0);
    mocker(RaGetSocketListenInfo, 10, 0);

    (void)RaSocketListenStopAsync(conn, 1, (void**)&reqHandle);
    free(reqHandle);
    mocker_clean();
    return;
}

void TcRaSocketListenStopAsyncFail()
{
    struct RaRequestHandle* reqHandle = NULL;
    struct RaSocketHandle socketHandle = {0};
    struct SocketListenInfoT conn[1] = {0};
    int ret = 0;

    conn[0].socketHandle = (void*)&socketHandle;

    mocker_clean();
    mocker(RaInetPton, 10, 0);
    mocker(RaHdcSendMsgAsync, 10, 0);
    mocker(RaGetSocketListenInfo, 10, 0);

    ret = RaSocketListenStopAsync(NULL, 1, (void**)&reqHandle);
    EXPECT_INT_NE(ret, 0);

    ret = RaSocketListenStopAsync(conn, 0, (void**)&reqHandle);
    EXPECT_INT_NE(ret, 0);

    ret = RaSocketListenStopAsync(conn, 1, NULL);
    EXPECT_INT_NE(ret, 0);

    conn[0].socketHandle = NULL;
    ret = RaSocketListenStopAsync(conn, 1, (void**)&reqHandle);
    EXPECT_INT_NE(ret, 0);

    mocker_clean();
    return;
}

void TcRaSocketListenStopAsync()
{
    TcRaSocketListenStopAsyncNormal();
    TcRaSocketListenStopAsyncFail();
}

void TcRaSocketBatchCloseAsyncNormal()
{
    struct RaRequestHandle* reqHandle = NULL;
    struct RaSocketHandle socketHandle = {0};
    struct SocketHdcInfo fdHandle = {0};
    struct SocketCloseInfoT conn[1] = {0};
    int ret = 0;

    conn[0].socketHandle = (void*)&socketHandle;
    conn[0].fdHandle = (void*)&fdHandle;

    mocker_clean();
    mocker(RaInetPton, 10, 0);
    mocker(RaHdcSendMsgAsync, 10, 0);
    mocker(RaGetSocketConnectInfo, 10, 0);

    ret = RaSocketBatchCloseAsync(conn, 1, (void**)&reqHandle);
    free(reqHandle->privData);
    free(reqHandle);
    mocker_clean();
    return;
}

void TcRaSocketBatchCloseAsyncFail()
{
    struct RaRequestHandle* reqHandle = NULL;
    struct RaSocketHandle socketHandle = {0};
    struct SocketHdcInfo fdHandle = {0};
    struct SocketCloseInfoT conn[1] = {0};
    int ret = 0;

    conn[0].socketHandle = (void*)&socketHandle;
    conn[0].fdHandle = (void*)&fdHandle;

    mocker_clean();
    mocker(RaInetPton, 10, 0);
    mocker(RaHdcSendMsgAsync, 10, 0);
    mocker(RaGetSocketConnectInfo, 10, 0);

    ret = RaSocketBatchCloseAsync(NULL, 1, (void**)&reqHandle);
    EXPECT_INT_NE(ret, 0);

    ret = RaSocketBatchCloseAsync(conn, 0, (void**)&reqHandle);
    EXPECT_INT_NE(ret, 0);

    ret = RaSocketBatchCloseAsync(conn, 1, NULL);
    EXPECT_INT_NE(ret, 0);

    conn[0].socketHandle = NULL;
    ret = RaSocketBatchCloseAsync(conn, 1, (void**)&reqHandle);
    EXPECT_INT_NE(ret, 0);

    mocker_clean();
    return;
}

void TcRaSocketBatchCloseAsync()
{
    TcRaSocketBatchCloseAsyncNormal();
    TcRaSocketBatchCloseAsyncFail();
}

void TcRaHdcAsyncInitSession()
{
    unsigned int connectStatus = HDC_CONNECTED;
    struct RaInitConfig cfg = {0};
    unsigned int phyId = 0;
    int ret = 0;

    mocker(pthread_create, 2, 0);
    mocker(DlDrvDeviceGetBareTgid, 1, 0);
    mocker(RaHdcAsyncMutexInit, 1, -1);
    RaHwAsyncSetConnectStatus(phyId, connectStatus);
    ret = RaHdcAsyncInitSession(&cfg);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRaGetEidByIpAsync()
{
    struct RaRequestHandle* reqHandle = NULL;
    struct RaCtxHandle ctxHandle = {0};
    union HccpEid eid[32] = {0};
    struct IpInfo ip[32] = {0};
    unsigned int num = 32;
    int ret = 0;

    mocker_clean();
    ret = RaGetEidByIpAsync(NULL, ip, eid, &num, (void**)&reqHandle);
    EXPECT_INT_EQ(ret, 128103);

    num = 33;
    ret = RaGetEidByIpAsync(&ctxHandle, ip, eid, &num, (void**)&reqHandle);
    EXPECT_INT_EQ(ret, 128103);

    num = 32;
    mocker(RaHdcGetEidByIpAsync, 10, -1);
    ret = RaGetEidByIpAsync(&ctxHandle, ip, eid, &num, (void**)&reqHandle);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(RaHdcGetEidByIpAsync, 10, 0);
    ret = RaGetEidByIpAsync(&ctxHandle, ip, eid, &num, (void**)&reqHandle);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRaHdcGetEidByIpAsync()
{
    union OpGetEidByIpData* asyncData = NULL;
    struct RaResponseEidList* privData = NULL;
    struct RaRequestHandle* reqHandle = NULL;
    struct RaCtxHandle ctxHandle = {0};
    unsigned int privDataNum = 1;
    union HccpEid eid[32] = {0};
    struct IpInfo ip[32] = {0};
    unsigned int num = 32;
    int ret = 0;

    mocker_clean();
    mocker(calloc, 1, NULL);
    ret = RaHdcGetEidByIpAsync(&ctxHandle, ip, eid, &num, (void**)&reqHandle);
    EXPECT_INT_EQ(ret, -ENOMEM);
    mocker_clean();

    mocker_invoke(calloc, CallocFirstStub, 2);
    ret = RaHdcGetEidByIpAsync(&ctxHandle, ip, eid, &num, (void**)&reqHandle);
    EXPECT_INT_EQ(ret, -ENOMEM);
    mocker_clean();

    mocker(RaHdcSendMsgAsync, 1, -1);
    ret = RaHdcGetEidByIpAsync(&ctxHandle, ip, eid, &num, (void**)&reqHandle);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(RaHdcSendMsgAsync, 1, 0);
    ret = RaHdcGetEidByIpAsync(&ctxHandle, ip, eid, &num, (void**)&reqHandle);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker(memcpy_s, 1, 0);
    asyncData = calloc(1, sizeof(union OpGetEidByIpData));
    reqHandle->opRet = 0;
    asyncData->rxData.num = 1;
    privData = (struct RaResponseEidList*)(reqHandle->privData);
    privData->num = &privDataNum;
    reqHandle->recvBuf = (void*)asyncData;
    RaHdcAsyncHandleGetEidByIp(reqHandle);
    mocker_clean();

    free(asyncData);
    asyncData = NULL;
    free(reqHandle);
    reqHandle = NULL;
}

void TcRaGetIpByEidAsync()
{
    struct RaRequestHandle* reqHandle = NULL;
    struct RaCtxHandle ctxHandle = {0};
    union HccpEid eid[32] = {0};
    struct IpInfo ip[32] = {0};
    unsigned int num = 32;
    int ret = 0;

    mocker_clean();
    ret = RaGetIpByEidAsync(NULL, eid, ip, &num, (void**)&reqHandle);
    EXPECT_INT_EQ(ret, 128103);

    num = 33;
    ret = RaGetIpByEidAsync(&ctxHandle, eid, ip, &num, (void**)&reqHandle);
    EXPECT_INT_EQ(ret, 128103);

    num = 32;
    mocker(RaHdcGetIpByEidAsync, 10, -1);
    ret = RaGetIpByEidAsync(&ctxHandle, eid, ip, &num, (void**)&reqHandle);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(RaHdcGetIpByEidAsync, 10, 0);
    ret = RaGetIpByEidAsync(&ctxHandle, eid, ip, &num, (void**)&reqHandle);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRaHdcGetIpByEidAsync()
{
    union OpGetIpByEidData* asyncData = NULL;
    struct RaResponseIpList* privData = NULL;
    struct RaRequestHandle* reqHandle = NULL;
    struct RaCtxHandle ctxHandle = {0};
    unsigned int privDataNum = 1;
    union HccpEid eid[32] = {0};
    struct IpInfo ip[32] = {0};
    unsigned int num = 32;
    int ret = 0;

    mocker_clean();
    mocker(calloc, 1, NULL);
    ret = RaHdcGetIpByEidAsync(&ctxHandle, eid, ip, &num, (void**)&reqHandle);
    EXPECT_INT_EQ(ret, -ENOMEM);
    mocker_clean();

    mocker_invoke(calloc, CallocFirstStub, 2);
    ret = RaHdcGetIpByEidAsync(&ctxHandle, eid, ip, &num, (void**)&reqHandle);
    EXPECT_INT_EQ(ret, -ENOMEM);
    mocker_clean();

    mocker(RaHdcSendMsgAsync, 1, -1);
    ret = RaHdcGetIpByEidAsync(&ctxHandle, eid, ip, &num, (void**)&reqHandle);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(RaHdcSendMsgAsync, 1, 0);
    ret = RaHdcGetIpByEidAsync(&ctxHandle, eid, ip, &num, (void**)&reqHandle);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker(memcpy_s, 1, 0);
    asyncData = calloc(1, sizeof(union OpGetIpByEidData));
    reqHandle->opRet = 0;
    asyncData->rxData.num = 1;
    privData = (struct RaResponseIpList*)(reqHandle->privData);
    privData->num = &privDataNum;
    reqHandle->recvBuf = (void*)asyncData;
    RaHdcAsyncHandleGetIpByEid(reqHandle);
    mocker_clean();

    free(asyncData);
    asyncData = NULL;
    free(reqHandle);
    reqHandle = NULL;
}

void TcRaHdcAsyncSessionClose()
{
    int ret = 0;

    mocker_clean();
    pthread_mutex_init(&gRaHdcAsync[0].reqMutex, NULL);
    mocker_invoke(RaHdcSendMsgAsync, RaHdcSendMsgAsyncStub, 1);
    mocker(RaHdcProcessMsg, 1, 0);
    mocker_invoke(HdcAsyncDelResponse, HdcAsyncDelResponseStub, 1);
    ret = RaHdcAsyncSessionClose(0);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    pthread_mutex_destroy(&gRaHdcAsync[0].reqMutex);
}
