/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ut_dispatch.h"
#include <stdlib.h>
#include <errno.h>
#include "securec.h"
#include "ra_hdc.h"
#include "ra_hdc_rdma_notify.h"
#include "ra_hdc_rdma.h"
#include "ra_hdc_socket.h"
#include "ra_hdc_lite.h"
#include "ra_hdc_tlv.h"
#include "hccp.h"
#include "ra_comm.h"
#include "ra_rs_err.h"

#define RA_QP_TX_DEPTH 32767
#define TC_TLV_HDC_MSG_SIZE (32 * 1024)
typedef uint32_t u32;
typedef uint16_t u16;
typedef unsigned long long u64;
typedef signed int s32;

extern int RaHdcInitApart(unsigned int phyId, unsigned int* logicId);
extern int RaHdcSessionClose(unsigned int phyId);
extern int RaHdcProcessMsg(unsigned int opcode, unsigned int deviceId, char* data, unsigned int dataSize);
extern int RaHdcNotifyCfgSet(unsigned int phyId, unsigned long long va, unsigned long long size);
extern int RaHdcNotifyCfgGet(unsigned int phyId, unsigned long long* va, unsigned long long* size);
extern STATIC int RaHdcNotifyBaseAddrInit(unsigned int notifyType, unsigned int phyId, unsigned long long** notifyVa);
extern void RaHdcLiteSaveCqeErrInfo(struct RaQpHandle* qpHdc, unsigned int status);
extern struct rdma_lite_context* RaRdmaLiteAllocCtx(u8 phyId, struct dev_cap_info* cap);
extern struct rdma_lite_cq* RaRdmaLiteCreateCq(struct rdma_lite_context* liteCtx, struct rdma_lite_cq_attr* liteCqAttr);
extern struct rdma_lite_qp* RaRdmaLiteCreateQp(struct rdma_lite_context* liteCtx, struct rdma_lite_qp_attr* liteQpAttr);
extern int RaHdcGetLiteSupport(struct RaRdmaHandle* rdmaHandle, unsigned int phyId);
extern int RaHdcGetDrvLiteSupport(unsigned int phyId, bool enabled910aLite, unsigned int* support);
extern int DlDrvDeviceGetIndexByPhyId(uint32_t phyId, uint32_t* devIndex);
extern int DlHalMemCtl(int type, void* paramValue, size_t paramValueSize, void* outValue, size_t* outSizeRet);
extern int RaRdmaLiteSetQpSl(struct rdma_lite_qp* liteQp, unsigned char sl);
extern int RaHdcLitePostSend(
    struct RaQpHandle* qpHdc, struct LiteMrInfo* localMr, struct LiteMrInfo* remMr, struct LiteSendWr* wr,
    struct SendWrRsp* wrRsp);
extern void RaHdcGetOpcodeLiteSupport(unsigned int phyId, unsigned int supportFeature, int* support);
extern int RaHdcGetOpcodeVersion(unsigned int phyId, unsigned int interfaceOpcode, unsigned int* interfaceVersion);
extern void RaHdcLiteMutexDeinit(struct RaRdmaHandle* rdmaHandle);
extern void RaRdmaLiteFreeCtx(struct rdma_lite_context* liteCtx);
extern int DlHalSensorNodeUnregister(uint32_t devid, uint64_t handle);
extern int RaSensorNodeRegister(unsigned int phyId, struct RaRdmaHandle* rdmaHandle);
extern int RaHdcLiteMutexInit(struct RaRdmaHandle* rdmaHandle, unsigned int phyId);
extern int RaHdcLiteGetCqQpAttr(
    struct RaQpHandle* qpHdc, struct rdma_lite_cq_attr* liteSendCqAttr, struct rdma_lite_cq_attr* liteRecvCqAttr,
    struct rdma_lite_qp_attr* liteQpAttr);
extern int RaHdcLiteInitMemPool(
    struct RaRdmaHandle* rdmaHandle, struct RaQpHandle* qpHdc, struct rdma_lite_cq_attr* liteSendCqAttr,
    struct rdma_lite_cq_attr* liteRecvCqAttr, struct rdma_lite_qp_attr* liteQpAttr);
extern int RaRdmaLiteDestroyCq(struct rdma_lite_cq* liteCq);
extern void RaHdcLiteDeinitMemPool(struct RaRdmaHandle* rdmaHandle, struct RaQpHandle* qpHdc);
extern void
RaHdcLiteQpAttrInit(struct RaQpHandle* qpHdc, struct rdma_lite_qp_attr* liteQpAttr, struct rdma_lite_qp_cap* cap);
extern int RaRdmaLiteInitMemPool(struct rdma_lite_context* liteCtx, struct rdma_lite_mem_attr* liteMemAttr);
extern int RaRdmaLiteDeinitMemPool(struct rdma_lite_context* liteCtx, u32 memIdx);
extern int RaHdcGetCqeErrInfoNum(struct RaRdmaHandle* rdmaHandle, unsigned int* num);
extern int RaRdmaLiteDestroyQp(struct rdma_lite_qp* liteQp);
extern int RaHdcSendPid(unsigned int phyId, struct ProcessRaSign pRaSign);
extern int RaHdcGetTlvRecvMsg(struct TlvMsg* recvMsg, char* recvData);
extern int RaRdmaLiteRestoreSnapshot(struct rdma_lite_context* liteCtx);
extern unsigned int RaRdmaLiteGetApiVersion(void);
extern void RaHdcLitePeriodPollCqe(struct RaRdmaHandle* rdmaHandle);
extern int RaHdcLiteCtxInit(struct RaRdmaHandle* rdmaHandle, unsigned int phyId, unsigned int rdevIndex);
extern int RaIsFirstUsed(int insId);
extern int RaIsLastUsed(int insId);

struct MsgHead gMsgTmp;
DLLEXPORT hdcError_t stub_drvHdcGetMsgBuffer_pid_error(struct drvHdcMsg* msg, int index, char** pBuf, int* pLen)
{
    gMsgTmp.ret = -EPERM;
    *pBuf = (char*)&gMsgTmp;
    *pLen = ((struct drvHdcMsgBuf*)(msg + 1))->len;
    return DRV_ERROR_NONE;
}

static unsigned int gDevid = 0;
static struct RaInitConfig gConfig = {
    .phyId = 0,
    .nicPosition = NETWORK_OFFLINE,
    .hdcType = HDC_SERVICE_TYPE_RDMA,
};
DLLEXPORT drvError_t StubSessionConnect(int peerNode, int peerDevid, HDC_CLIENT client, HDC_SESSION* session)
{
    static HDC_SESSION gHdcSession = (HDC_SESSION)1;
    *session = gHdcSession;
    return 0;
}
static char gDrvRecvMsg[4096];
static int gDrvRecvMsgLen = 0;
static DLLEXPORT drvError_t stub_drvHdcGetMsgBuffer(struct drvHdcMsg* msg, int index, char** pBuf, int* pLen)
{
    *pBuf = gDrvRecvMsg;
    *pLen = gDrvRecvMsgLen;
    return 0;
}

DLLEXPORT drvError_t drvGetDevNum(unsigned int* numDev);

int RaHdcGetAllVnic(unsigned int* vnicIp, unsigned int num);

unsigned int gInterfaceVersion = 0;

int StubRaGetInterfaceVersion(unsigned int phyId, unsigned int interfaceOpcode, unsigned int* interfaceVersion)
{
    *interfaceVersion = gInterfaceVersion;
    return 0;
}

void TcHdcInit()
{
    struct ProcessRaSign pRaSign;
    pRaSign.tgid = 0;
    struct RaInitConfig config = {
        .phyId = gDevid,
        .nicPosition = NETWORK_OFFLINE,
        .hdcType = HDC_SERVICE_TYPE_RDMA,
    };
    mocker((stub_fn_t)drvHdcClientCreate, 1, 0);
    mocker_invoke((stub_fn_t)drvHdcSessionConnect, (stub_fn_t)StubSessionConnect, 1);
    mocker((stub_fn_t)drvHdcSetSessionReference, 1, 0);
    mocker((stub_fn_t)halHdcRecv, 2, 0);
    int ret = RaHdcInit(&config, pRaSign);
    EXPECT_INT_EQ(ret, 0);

    mocker((stub_fn_t)memset_s, 1, 0);
    ret = RaHdcInit(&config, pRaSign);
    EXPECT_INT_EQ(ret, -EEXIST);

    mocker((stub_fn_t)drvHdcSessionClose, 1, 0);
    mocker((stub_fn_t)drvHdcClientDestroy, 1, 0);
    ret = RaHdcDeinit(&config);
    EXPECT_INT_EQ(ret, 0);
}

void TcHdcTestEnvInit()
{
    struct ProcessRaSign pRaSign;
    pRaSign.tgid = 0;

    mocker((stub_fn_t)drvHdcClientCreate, 1, 0);
    mocker_invoke((stub_fn_t)drvHdcSessionConnect, (stub_fn_t)StubSessionConnect, 1);
    mocker((stub_fn_t)drvHdcSetSessionReference, 1, 0);
    mocker((stub_fn_t)halHdcRecv, 10, 0);
    int ret = RaHdcInit(&gConfig, pRaSign);
    EXPECT_INT_EQ(ret, 0);
}

void TcHdcTestEnvDeinit()
{
    mocker((stub_fn_t)drvHdcSessionClose, 1, 0);
    mocker((stub_fn_t)drvHdcClientDestroy, 1, 0);
    int ret = RaHdcDeinit(&gConfig);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcHdcInitFail()
{
    mocker_clean();
    struct ProcessRaSign pRaSign;
    pRaSign.tgid = 0;
    struct RaInitConfig config = {
        .phyId = RA_MAX_PHY_ID_NUM,
        .nicPosition = NETWORK_OFFLINE,
        .hdcType = HDC_SERVICE_TYPE_RDMA,
    };
    mocker(RaHdcInitApart, 1, -1);
    int ret = RaHdcInit(&config, pRaSign);
    EXPECT_INT_NE(ret, 0);

    config.phyId = gDevid;

    mocker_clean();
    mocker((stub_fn_t)RaHdcInitApart, 1, -EINVAL);
    ret = RaHdcInit(&config, pRaSign);
    EXPECT_INT_EQ(ret, -EINVAL);

    mocker_clean();
    mocker((stub_fn_t)pthread_mutex_init, 1, -1);
    ret = RaHdcInit(&config, pRaSign);
    EXPECT_INT_EQ(ret, -ESYSFUNC);

    mocker_clean();
    mocker((stub_fn_t)drvHdcClientCreate, 1, 0);
    mocker((stub_fn_t)drvHdcSessionConnect, 1, -1);
    ret = RaHdcInit(&config, pRaSign);
    EXPECT_INT_EQ(ret, -EPERM);

    mocker_clean();
    mocker((stub_fn_t)drvHdcClientCreate, 1, 0);
    mocker_invoke((stub_fn_t)drvHdcSessionConnect, (stub_fn_t)StubSessionConnect, 1);
    mocker((stub_fn_t)drvHdcSetSessionReference, 1, -1);
    ret = RaHdcInit(&config, pRaSign);
    EXPECT_INT_EQ(ret, -EPERM);
}

void TcHdcDeinitFail()
{
    struct RaInitConfig config = {
        .phyId = RA_MAX_PHY_ID_NUM,
        .nicPosition = NETWORK_OFFLINE,
        .hdcType = HDC_SERVICE_TYPE_RDMA,
    };
    int ret;

    mocker_clean();
    mocker((stub_fn_t)drvHdcSessionClose, 10, -10);
    mocker((stub_fn_t)drvHdcClientDestroy, 10, -10);
    mocker((stub_fn_t)pthread_mutex_destroy, 10, -10);
    mocker((stub_fn_t)calloc, 1, NULL);
    config.phyId = gDevid;
    ret = RaHdcDeinit(&config);
    EXPECT_INT_EQ(ret, -ESYSFUNC);
    mocker_clean();
}

void TcHdcSocketBatchConnect()
{
    struct SocketConnectInfoT conn[1] = {{0}};
    mocker_clean();
    TcHdcTestEnvInit();

    mocker((stub_fn_t)RaGetSocketConnectInfo, 10, 0);
    int ret = RaHdcSocketBatchConnect(gDevid, conn, 1);
    EXPECT_INT_EQ(ret, 0);

    TcHdcTestEnvDeinit();

    mocker_clean();
    mocker((stub_fn_t)calloc, 1, NULL);
    ret = RaHdcSocketBatchConnect(gDevid, conn, 1);
    EXPECT_INT_EQ(ret, -ENOMEM);

    mocker_clean();
    mocker((stub_fn_t)RaGetSocketConnectInfo, 1, -1);
    ret = RaHdcSocketBatchConnect(gDevid, conn, 1);
    EXPECT_INT_EQ(ret, -1);

    mocker_clean();
    mocker((stub_fn_t)RaGetSocketConnectInfo, 1, 0);
    ret = RaHdcSocketBatchConnect(0, conn, 1);
    EXPECT_INT_EQ(ret, -22);

    mocker_clean();
}

void TcHdcSocketBatchClose()
{
    struct SocketCloseInfoT conn[1] = {0};
    conn[0].fdHandle = calloc(sizeof(struct SocketHdcInfo), 1);
    ((struct SocketHdcInfo*)conn[0].fdHandle)->fd = 0;
    ((struct SocketHdcInfo*)conn[0].fdHandle)->phyId = 0;
    mocker_clean();
    TcHdcTestEnvInit();
    int ret = RaHdcSocketBatchClose(gDevid, conn, 1);
    EXPECT_INT_EQ(ret, 0);

    TcHdcTestEnvDeinit();

    conn[0].fdHandle = calloc(sizeof(struct SocketHdcInfo), 1);
    ((struct SocketHdcInfo*)conn[0].fdHandle)->fd = 0;
    ((struct SocketHdcInfo*)conn[0].fdHandle)->phyId = 0;
    mocker_clean();
    mocker((stub_fn_t)calloc, 10, NULL);
    ret = RaHdcSocketBatchClose(gDevid, conn, 1);
    EXPECT_INT_EQ(ret, -ENOMEM);
    mocker_clean();

    conn[0].fdHandle = calloc(sizeof(struct SocketHdcInfo), 1);
    ((struct SocketHdcInfo*)conn[0].fdHandle)->fd = 0;
    ((struct SocketHdcInfo*)conn[0].fdHandle)->phyId = 0;
    mocker((stub_fn_t)RaHdcProcessMsg, 10, -1);
    ret = RaHdcSocketBatchClose(gDevid, conn, 1);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcHdcSocketListenStart()
{
    int ret;
    struct RaSocketHandle socketHandle = {0};
    struct SocketListenInfoT conn[1] = {0};
    conn[0].socketHandle = &socketHandle;

    mocker_clean();
    TcHdcTestEnvInit();
    mocker((stub_fn_t)RaHdcProcessMsg, 5, 0);
    mocker((stub_fn_t)RaGetSocketListenResult, 10, 0);
    ret = RaHdcSocketListenStart(gDevid, conn, 1);
    EXPECT_INT_EQ(ret, 0);

    TcHdcTestEnvDeinit();

    mocker_clean();
    mocker((stub_fn_t)RaGetSocketListenInfo, 1, -1);
    ret = RaHdcSocketListenStart(gDevid, conn, 1);
    EXPECT_INT_EQ(ret, -EINVAL);

    mocker_clean();
    mocker((stub_fn_t)RaGetSocketListenInfo, 1, 0);
    ret = RaHdcSocketListenStart(0, conn, 1);
    EXPECT_INT_EQ(ret, -EINVAL);

    mocker_clean();
    mocker((stub_fn_t)RaHdcProcessMsg, 1, 0);
    mocker((stub_fn_t)RaGetSocketListenResult, 10, -1);
    ret = RaHdcSocketListenStart(gDevid, conn, 1);
    EXPECT_INT_EQ(ret, -EINVAL);

    mocker_clean();
    mocker((stub_fn_t)RaGetSocketConnectInfo, 1, -1);
    ret = RaHdcSocketListenStart(gDevid, conn, 1);
    EXPECT_INT_EQ(ret, -EINVAL);
    mocker_clean();
}

void TcHdcSocketBatchAbort()
{
    struct SocketConnectInfoT conn[1] = {0};
    struct RaSocketHandle socketHandle = {0};
    conn[0].socketHandle = &socketHandle;

    mocker_clean();
    TcHdcTestEnvInit();

    mocker((stub_fn_t)RaGetSocketConnectInfo, 10, 0);
    int ret = RaHdcSocketBatchAbort(gDevid, conn, 1);
    EXPECT_INT_EQ(ret, 0);

    TcHdcTestEnvDeinit();

    mocker_clean();
    mocker((stub_fn_t)calloc, 1, NULL);
    ret = RaHdcSocketBatchAbort(gDevid, conn, 1);
    EXPECT_INT_EQ(ret, -ENOMEM);

    mocker_clean();
    mocker((stub_fn_t)RaGetSocketConnectInfo, 1, -1);
    ret = RaHdcSocketBatchAbort(gDevid, conn, 1);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcHdcSocketListenStop()
{
    struct SocketListenInfoT conn[1] = {{0}};
    struct RaSocketHandle socketHandle = {0};
    conn[0].socketHandle = &socketHandle;

    mocker_clean();
    TcHdcTestEnvInit();
    int ret = RaHdcSocketListenStop(gDevid, conn, 1);
    EXPECT_INT_EQ(ret, 0);

    TcHdcTestEnvDeinit();

    mocker_clean();
    mocker((stub_fn_t)RaGetSocketConnectInfo, 1, -1);
    ret = RaHdcSocketListenStart(gDevid, conn, 1);
    EXPECT_INT_EQ(ret, -EINVAL);
    mocker_clean();

    mocker((stub_fn_t)RaGetSocketListenInfo, 1, -1);
    ret = RaHdcSocketListenStop(gDevid, conn, 1);
    EXPECT_INT_EQ(ret, -EINVAL);
    mocker_clean();

    mocker((stub_fn_t)RaHdcProcessMsg, 1, -1);
    ret = RaHdcSocketListenStop(gDevid, conn, 1);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcHdcGetSockets()
{
    struct SocketInfoT conn[1];
    conn[0].fdHandle = NULL;
    conn[0].socketHandle = calloc(sizeof(struct RaSocketHandle), 1);

    mocker_clean();
    TcHdcTestEnvInit();
    int ret = RaHdcGetSockets(gDevid, 0, conn, 1);
    EXPECT_INT_NE(ret, 0);
    TcHdcTestEnvDeinit();

    mocker_clean();
    mocker((stub_fn_t)calloc, 1, NULL);
    ret = RaHdcGetSockets(gDevid, 0, conn, 1);
    EXPECT_INT_EQ(ret, -ENOMEM);

    mocker_clean();
    mocker((stub_fn_t)RaHdcProcessMsg, 1, -1);
    ret = RaHdcGetSockets(gDevid, 0, conn, 1);
    EXPECT_INT_EQ(ret, -EINVAL);

    mocker_clean();
    mocker((stub_fn_t)memcpy_s, 1, -1);
    ret = RaHdcGetSockets(gDevid, 0, conn, 1);
    EXPECT_INT_EQ(ret, -EINVAL);
    mocker_clean();

    free(conn[0].socketHandle);
    if (conn[0].fdHandle != NULL) {
        free(conn[0].fdHandle);
        conn[0].fdHandle = NULL;
    }
}

void TcHdcSocketSend()
{
    struct SocketHdcInfo socketInfo = {0};
    char sendBuf[16] = {0};

    mocker_clean();
    TcHdcTestEnvInit();
    mocker_invoke((stub_fn_t)drvHdcGetMsgBuffer, (stub_fn_t)stub_drvHdcGetMsgBuffer, 10);
    struct MsgHead* testHead = (struct MsgHead*)gDrvRecvMsg;
    gDrvRecvMsgLen = sizeof(union OpSocketSendData) + sizeof(struct MsgHead);
    testHead->opcode = RA_RS_SOCKET_SEND;
    testHead->msgDataLen = sizeof(union OpSocketSendData);
    testHead->ret = 0;

    union OpSocketSendData* data = (union OpSocketSendData*)(gDrvRecvMsg + sizeof(struct MsgHead));
    data->rxData.realSendSize = 100;

    int ret = RaHdcSocketSend(gDevid, &socketInfo, sendBuf, sizeof(sendBuf));
    EXPECT_INT_EQ(ret, 100);

    testHead->ret = -EAGAIN;
    ret = RaHdcSocketSend(gDevid, &socketInfo, sendBuf, sizeof(sendBuf));
    EXPECT_INT_EQ(ret, -EAGAIN);

    mocker((stub_fn_t)drvHdcAddMsgBuffer, 10, 10);
    ret = RaHdcSocketSend(gDevid, &socketInfo, sendBuf, sizeof(sendBuf));
    EXPECT_INT_EQ(ret, -10);
    mocker((stub_fn_t)RaHdcSessionClose, 10, 0);
    TcHdcTestEnvDeinit();

    char maxSendBuf[SOCKET_SEND_MAXLEN + 1] = {0};
    mocker_clean();
    mocker((stub_fn_t)calloc, 1, NULL);
    ret = RaHdcSocketSend(gDevid, &socketInfo, maxSendBuf, sizeof(maxSendBuf));
    EXPECT_INT_EQ(ret, -ENOMEM);

    mocker_clean();
    mocker((stub_fn_t)memcpy_s, 1, -1);
    ret = RaHdcSocketSend(gDevid, &socketInfo, maxSendBuf, sizeof(maxSendBuf));
    EXPECT_INT_EQ(ret, -ESAFEFUNC);
}

void TcHdcSocketRecv()
{
    struct SocketHdcInfo socketInfo = {0};
    char sendBuf[16] = {0};

    mocker_clean();
    TcHdcTestEnvInit();
    mocker((stub_fn_t)RaHdcProcessMsg, 5, 0);
    mocker((stub_fn_t)memcpy_s, 10, 0);
    int ret = RaHdcSocketRecv(gDevid, &socketInfo, sendBuf, sizeof(sendBuf));
    TcHdcTestEnvDeinit();

    mocker_clean();
    mocker((stub_fn_t)calloc, 1, NULL);
    ret = RaHdcSocketRecv(gDevid, &socketInfo, sendBuf, sizeof(sendBuf));
    EXPECT_INT_EQ(ret, -ENOMEM);

    mocker_clean();
    mocker((stub_fn_t)RaHdcProcessMsg, 1, -1);
    ret = RaHdcSocketRecv(gDevid, &socketInfo, sendBuf, sizeof(sendBuf));
    EXPECT_INT_EQ(ret, -1);

    mocker_clean();
    mocker((stub_fn_t)RaHdcProcessMsg, 5, 0);
    mocker((stub_fn_t)memcpy_s, 10, -1);
    ret = RaHdcSocketRecv(gDevid, &socketInfo, sendBuf, sizeof(sendBuf));
}

void TcHdcQpCreateDestroy()
{
    mocker_clean();
    TcHdcTestEnvInit();
    struct RaRdmaHandle rdmaHandle = {0};
    void* qpHandle = NULL;
    rdmaHandle.supportLite = 1;
    int ret = RaHdcQpCreate(&rdmaHandle, 0, 0, &qpHandle);
    ASSERT_ADDR_NE(qpHandle, NULL);
    ret = RaHdcQpDestroy(qpHandle);
    EXPECT_INT_EQ(ret, 0);
    TcHdcTestEnvDeinit();

    mocker_clean();
    mocker((stub_fn_t)calloc, 1, NULL);
    ret = RaHdcQpCreate(&rdmaHandle, 0, 0, &qpHandle);
    EXPECT_INT_EQ(ret, -ENOMEM);
    mocker_clean();

    mocker((stub_fn_t)RaHdcProcessMsg, 1, -1);
    ret = RaHdcQpCreate(&rdmaHandle, 0, 0, &qpHandle);
    EXPECT_INT_EQ(ret, -1);
}

void TcHdcGetQpStatus()
{
    mocker_clean();
    TcHdcTestEnvInit();
    struct RaRdmaHandle rdmaHandle = {0};
    void* qpHandle = NULL;
    rdmaHandle.supportLite = 1;
    int ret = RaHdcQpCreate(&rdmaHandle, 0, 0, &qpHandle);
    ASSERT_ADDR_NE(qpHandle, NULL);
    int status = 0;
    ret = RaHdcGetQpStatus(qpHandle, &status);
    EXPECT_INT_EQ(ret, 0);
    ret = RaHdcQpDestroy(qpHandle);
    EXPECT_INT_EQ(ret, 0);
    TcHdcTestEnvDeinit();

    mocker_clean();
    struct RaQpHandle testQpHandle = {0};
    testQpHandle.phyId = gDevid;
    mocker((stub_fn_t)calloc, 1, NULL);
    ret = RaHdcGetQpStatus(&testQpHandle, &status);
    EXPECT_INT_NE(ret, 0);
}

void TcHdcQpConnectAsync()
{
    mocker_clean();
    TcHdcTestEnvInit();
    struct RaRdmaHandle rdmaHandle = {0};
    void* qpHandle = NULL;
    int ret = RaHdcQpCreate(&rdmaHandle, 0, 0, &qpHandle);
    struct SocketHdcInfo socketHandle = {0};
    ASSERT_ADDR_NE(qpHandle, NULL);

    ret = RaHdcQpConnectAsync(qpHandle, &socketHandle);
    EXPECT_INT_EQ(ret, 0);
    ret = RaHdcQpDestroy(qpHandle);
    EXPECT_INT_EQ(ret, 0);
    TcHdcTestEnvDeinit();

    mocker_clean();
    mocker((stub_fn_t)calloc, 1, NULL);
    struct RaQpHandle testQpHandle = {0};
    ret = RaHdcQpConnectAsync(&testQpHandle, &socketHandle);
    EXPECT_INT_NE(ret, 0);
}

void TcHdcMrDereg()
{
    mocker_clean();
    TcHdcTestEnvInit();
    struct RaRdmaHandle rdmaHandle = {0};
    void* qpHandle = NULL;
    int ret = RaHdcQpCreate(&rdmaHandle, 0, 0, &qpHandle);
    char qpReg[16] = {0};
    ASSERT_ADDR_NE(qpHandle, NULL);

    ret = RaHdcMrDereg(qpHandle, (struct MrInfoT*)qpReg);
    EXPECT_INT_EQ(ret, 0);
    ret = RaHdcQpDestroy(qpHandle);
    EXPECT_INT_EQ(ret, 0);
    TcHdcTestEnvDeinit();

    mocker_clean();
    mocker((stub_fn_t)calloc, 1, NULL);
    struct RaQpHandle testQpHandle = {0};
    ret = RaHdcMrDereg(&testQpHandle, (struct MrInfoT*)qpReg);
    EXPECT_INT_NE(ret, 0);
}

int StubRaHdcProcessMsgWrlist(unsigned int opcode, int deviceId, char* data, unsigned int dataSize)
{
    union OpSendWrlistData* sendWrlist = (union OpSendWrlistData*)data;
    sendWrlist->rxData.completeNum = 1;
    return 0;
}

int StubRaHdcProcessMsgWrlist1(unsigned int opcode, int deviceId, char* data, unsigned int dataSize)
{
    union OpSendWrlistData* sendWrlist = (union OpSendWrlistData*)data;
    sendWrlist->rxData.completeNum = 1;
    return -ENOENT;
}

int StubRaHdcProcessMsgWrlist2(unsigned int opcode, int deviceId, char* data, unsigned int dataSize)
{
    union OpSendWrlistData* sendWrlist = (union OpSendWrlistData*)data;
    sendWrlist->rxData.completeNum = 1;
    return -3;
}

int StubRaHdcProcessMsgWrlist3(unsigned int opcode, int deviceId, char* data, unsigned int dataSize)
{
    union OpSendWrlistData* sendWrlist = (union OpSendWrlistData*)data;
    sendWrlist->rxData.completeNum = 5;
    return 0;
}

void TcHdcSendWrlistV2()
{
    mocker_clean();
    struct SendWrlistData wrlist[1] = {{0}};
    struct SendWrRsp opRsp[1] = {{0}};
    struct WrlistSendCompleteNum wrlistNum = {0};
    unsigned int completeNum = 0;
    wrlistNum.sendNum = 1;
    wrlistNum.completeNum = &completeNum;

    struct RaQpHandle handle = {0};
    handle.qpMode = 1;

    mocker_invoke((stub_fn_t)RaHdcProcessMsg, (stub_fn_t)StubRaHdcProcessMsgWrlist, 1);
    gInterfaceVersion = RA_RS_SEND_WRLIST_V2_VERSION;
    mocker_invoke((stub_fn_t)RaGetInterfaceVersion, (stub_fn_t)StubRaGetInterfaceVersion, 1);
    int ret = RaHdcSendWrlist(&handle, wrlist, opRsp, wrlistNum);
    gInterfaceVersion = 0;
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker_invoke((stub_fn_t)RaHdcProcessMsg, (stub_fn_t)StubRaHdcProcessMsgWrlist1, 1);
    gInterfaceVersion = RA_RS_SEND_WRLIST_V2_VERSION;
    mocker_invoke((stub_fn_t)RaGetInterfaceVersion, (stub_fn_t)StubRaGetInterfaceVersion, 1);
    ret = RaHdcSendWrlist(&handle, wrlist, opRsp, wrlistNum);
    gInterfaceVersion = 0;
    EXPECT_INT_EQ(ret, -ENOENT);
    mocker_clean();

    mocker_invoke((stub_fn_t)RaHdcProcessMsg, (stub_fn_t)StubRaHdcProcessMsgWrlist2, 1);
    gInterfaceVersion = RA_RS_SEND_WRLIST_V2_VERSION;
    mocker_invoke((stub_fn_t)RaGetInterfaceVersion, (stub_fn_t)StubRaGetInterfaceVersion, 1);
    ret = RaHdcSendWrlist(&handle, wrlist, opRsp, wrlistNum);
    gInterfaceVersion = 0;
    EXPECT_INT_EQ(ret, -3);
    mocker_clean();

    mocker_invoke((stub_fn_t)RaHdcProcessMsg, (stub_fn_t)StubRaHdcProcessMsgWrlist3, 1);
    gInterfaceVersion = RA_RS_SEND_WRLIST_V2_VERSION;
    mocker_invoke((stub_fn_t)RaGetInterfaceVersion, (stub_fn_t)StubRaGetInterfaceVersion, 1);
    ret = RaHdcSendWrlist(&handle, wrlist, opRsp, wrlistNum);
    gInterfaceVersion = 0;
    EXPECT_INT_EQ(ret, -EINVAL);
    mocker_clean();

    mocker((stub_fn_t)memcpy_s, 1, -1);
    gInterfaceVersion = RA_RS_SEND_WRLIST_V2_VERSION;
    mocker_invoke((stub_fn_t)RaGetInterfaceVersion, (stub_fn_t)StubRaGetInterfaceVersion, 1);
    ret = RaHdcSendWrlist(&handle, wrlist, opRsp, wrlistNum);
    gInterfaceVersion = 0;
    EXPECT_INT_EQ(ret, -ESAFEFUNC);
    mocker_clean();

    mocker((stub_fn_t)calloc, 1, NULL);
    gInterfaceVersion = RA_RS_SEND_WRLIST_V2_VERSION;
    mocker_invoke((stub_fn_t)RaGetInterfaceVersion, (stub_fn_t)StubRaGetInterfaceVersion, 1);
    ret = RaHdcSendWrlist(&handle, wrlist, opRsp, wrlistNum);
    gInterfaceVersion = 0;
    EXPECT_INT_EQ(ret, -ENOMEM);
    mocker_clean();
}

void TcHdcSendWrlist()
{
    mocker_clean();
    struct SendWrlistData wrlist[1] = {{0}};
    struct SendWrRsp opRsp[1] = {{0}};
    struct WrlistSendCompleteNum wrlistNum = {0};
    unsigned int completeNum = 0;
    wrlistNum.sendNum = 1;
    wrlistNum.completeNum = &completeNum;

    struct RaQpHandle handle = {0};
    handle.qpMode = 1;

    mocker_invoke((stub_fn_t)RaHdcProcessMsg, (stub_fn_t)StubRaHdcProcessMsgWrlist, 1);
    int ret = RaHdcSendWrlist(&handle, wrlist, opRsp, wrlistNum);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker_invoke((stub_fn_t)RaHdcProcessMsg, (stub_fn_t)StubRaHdcProcessMsgWrlist1, 1);
    ret = RaHdcSendWrlist(&handle, wrlist, opRsp, wrlistNum);
    EXPECT_INT_EQ(ret, -ENOENT);
    mocker_clean();

    mocker_invoke((stub_fn_t)RaHdcProcessMsg, (stub_fn_t)StubRaHdcProcessMsgWrlist2, 1);
    ret = RaHdcSendWrlist(&handle, wrlist, opRsp, wrlistNum);
    EXPECT_INT_EQ(ret, -3);
    mocker_clean();

    mocker_invoke((stub_fn_t)RaHdcProcessMsg, (stub_fn_t)StubRaHdcProcessMsgWrlist3, 1);
    ret = RaHdcSendWrlist(&handle, wrlist, opRsp, wrlistNum);
    EXPECT_INT_EQ(ret, -EINVAL);
    mocker_clean();

    mocker((stub_fn_t)memcpy_s, 1, -1);
    ret = RaHdcSendWrlist(&handle, wrlist, opRsp, wrlistNum);
    EXPECT_INT_EQ(ret, -ESAFEFUNC);
    mocker_clean();

    mocker((stub_fn_t)calloc, 1, NULL);
    ret = RaHdcSendWrlist(&handle, wrlist, opRsp, wrlistNum);
    EXPECT_INT_EQ(ret, -ENOMEM);
    mocker_clean();

    struct ProcessRaSign pRaSign = {0};
    mocker((stub_fn_t)strcpy_s, 1, -1);
    ret = RaHdcSendPid(0, pRaSign);
    EXPECT_INT_EQ(ret, -ESAFEFUNC);
    mocker_clean();

    mocker((stub_fn_t)RaHdcProcessMsg, 1, -1);
    ret = RaHdcSendPid(0, pRaSign);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker((stub_fn_t)RaHdcProcessMsg, 1, -1);
    ret = RaHdcNotifyCfgSet(0, 0, 0);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker((stub_fn_t)RaHdcProcessMsg, 1, -1);
    unsigned long long va = 0;
    unsigned long long size = 0;
    ret = RaHdcNotifyCfgGet(0, &va, &size);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker((stub_fn_t)drvDeviceGetIndexByPhyId, 1, -1);
    unsigned int logicId = 0;
    ret = RaHdcInitApart(0, &logicId);
    EXPECT_INT_EQ(ret, -ENODEV);
    mocker_clean();

    mocker((stub_fn_t)pthread_mutex_init, 1, -1);
    ret = RaHdcInitApart(0, &logicId);
    EXPECT_INT_EQ(ret, -ESYSFUNC);
    mocker_clean();

    mocker((stub_fn_t)memcpy_s, 1, -1);
    char data = 0;
    ret = RaHdcProcessMsg(0, 0, &data, 0);
    EXPECT_INT_EQ(ret, -ESAFEFUNC);
    mocker_clean();

    TcHdcSendWrlistV2();
}

void TcHdcSendWr()
{
    mocker_clean();
    TcHdcTestEnvInit();
    struct RaRdmaHandle rdmaHandle = {0};
    void* qpHandle = NULL;
    int ret = RaHdcQpCreate(&rdmaHandle, 0, 0, &qpHandle);
    struct SendWr wr = {0};
    struct SendWrRsp rsp = {0};

    ASSERT_ADDR_NE(qpHandle, NULL);

    ret = RaHdcSendWr(qpHandle, &wr, &rsp);
    EXPECT_INT_EQ(ret, 0);

    struct MsgHead* testHead = (struct MsgHead*)gDrvRecvMsg;
    testHead->ret = -ENOENT;
    testHead->opcode = RA_RS_SEND_WR;
    gDrvRecvMsgLen = sizeof(union OpSendWrData) + sizeof(struct MsgHead);
    mocker_invoke((stub_fn_t)drvHdcGetMsgBuffer, (stub_fn_t)stub_drvHdcGetMsgBuffer, 3);

    ret = RaHdcSendWr(qpHandle, &wr, &rsp);
    EXPECT_INT_EQ(ret, -ENOENT);

    testHead->ret = 0;
    testHead->opcode = RA_RS_QP_DESTROY;
    testHead->msgDataLen = sizeof(union OpQpDestroyData);
    gDrvRecvMsgLen = sizeof(union OpQpDestroyData) + sizeof(struct MsgHead);
    ret = RaHdcQpDestroy(qpHandle);
    EXPECT_INT_EQ(ret, 0);

    mocker_clean();
    mocker((stub_fn_t)drvHdcClientCreate, 1, 0);
    mocker_invoke((stub_fn_t)drvHdcSessionConnect, (stub_fn_t)StubSessionConnect, 1);
    mocker((stub_fn_t)drvHdcSetSessionReference, 1, 0);
    mocker((stub_fn_t)halHdcRecv, 10, 0);
    TcHdcTestEnvDeinit();

    struct RaQpHandle testQpHandle = {0};
    mocker_clean();
    mocker((stub_fn_t)memcpy_s, 1, -1);
    ret = RaHdcSendWr(&testQpHandle, &wr, &rsp);
    EXPECT_INT_EQ(ret, -ESAFEFUNC);

    mocker_clean();
    mocker((stub_fn_t)calloc, 1, NULL);
    ret = RaHdcSendWr(&testQpHandle, &wr, &rsp);
    EXPECT_INT_NE(ret, 0);
}

void TcHdcGetNotifyBaseAddr()
{
    mocker_clean();
    TcHdcTestEnvInit();
    struct RaRdmaHandle rdmaHandle = {0};
    void* qpHandle = NULL;
    int ret = RaHdcQpCreate(&rdmaHandle, 0, 0, &qpHandle);
    unsigned long long va;
    unsigned long long size;
    ASSERT_ADDR_NE(qpHandle, NULL);

    ret = RaHdcGetNotifyBaseAddr(qpHandle, &va, &size);
    EXPECT_INT_EQ(ret, 0);

    ret = RaHdcQpDestroy(qpHandle);
    EXPECT_INT_EQ(ret, 0);
    TcHdcTestEnvDeinit();

    struct RaQpHandle testQpHandle = {0};
    mocker_clean();
    mocker((stub_fn_t)calloc, 1, NULL);
    ret = RaHdcGetNotifyBaseAddr((struct RaRdmaHandle*)&testQpHandle, &va, &size);
    EXPECT_INT_NE(ret, 0);
}

void TcHdcSocketInit()
{
    mocker_clean();
    TcHdcTestEnvInit();
    struct rdev rdevInfo = {0};

    int ret = RaHdcSocketInit(rdevInfo);
    EXPECT_INT_EQ(ret, 0);
    TcHdcTestEnvDeinit();

    mocker_clean();
    mocker((stub_fn_t)RaHdcGetAllVnic, 1, 0);
    mocker((stub_fn_t)memcpy_s, 1, -10);
    ret = RaHdcSocketInit(rdevInfo);
    EXPECT_INT_EQ(ret, -ESAFEFUNC);

    mocker_clean();
    mocker((stub_fn_t)drvHdcAllocMsg, 1, -10);
    ret = RaHdcSocketInit(rdevInfo);
    EXPECT_INT_EQ(ret, -EINVAL);
    mocker_clean();

    mocker((stub_fn_t)memset_s, 1, -1);
    ret = RaHdcSocketInit(rdevInfo);
    EXPECT_INT_EQ(ret, -ESAFEFUNC);
    mocker_clean();

    mocker((stub_fn_t)drvGetDevNum, 1, 1);
    ret = RaHdcSocketInit(rdevInfo);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker((stub_fn_t)RaHdcGetAllVnic, 1, -1);
    ret = RaHdcSocketInit(rdevInfo);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcHdcSocketDeinit()
{
    mocker_clean();
    TcHdcTestEnvInit();
    struct rdev rdevInfo = {0};
    int ret = RaHdcSocketDeinit(rdevInfo);
    EXPECT_INT_EQ(ret, 0);
    TcHdcTestEnvDeinit();

    mocker_clean();
    mocker((stub_fn_t)memset_s, 1, -10);
    ret = RaHdcSocketDeinit(rdevInfo);
    EXPECT_INT_EQ(ret, -ESAFEFUNC);

    mocker_clean();
    mocker((stub_fn_t)memcpy_s, 1, -10);
    ret = RaHdcSocketDeinit(rdevInfo);
    EXPECT_INT_EQ(ret, -ESAFEFUNC);

    mocker_clean();
    mocker((stub_fn_t)drvHdcAllocMsg, 1, -10);
    ret = RaHdcSocketDeinit(rdevInfo);
    EXPECT_INT_EQ(ret, -EINVAL);
}

int StubRaHdcProcessRdevInit(unsigned int opcode, int deviceId, char* data, unsigned int dataSize)
{
    union OpRdevInitData* rdevInitData;
    union OpLiteSupportData* liteSupportData;

    if (opcode == RA_RS_RDEV_INIT) {
        rdevInitData = (union OpRdevInitData*)data;
        rdevInitData->rxData.rdevIndex = 0;
    } else if (opcode == RA_RS_GET_LITE_SUPPORT) {
        liteSupportData = (union OpLiteSupportData*)data;
        liteSupportData->rxData.supportLite = 1;
    }
    return 0;
}

void StubRaHdcGetOpcodeLiteSupport(unsigned int phyId, unsigned int supportFeature, int* support)
{
    if (supportFeature == 1) {
        *support = 1;
        return;
    }
}

int StubRaHdcProcessRdevInitError(unsigned int opcode, int deviceId, char* data, unsigned int dataSize)
{
    union OpRdevInitData* rdevInitData;

    if (opcode == RA_RS_RDEV_INIT) {
        rdevInitData = (union OpRdevInitData*)data;
        rdevInitData->rxData.rdevIndex = 0;
    } else if (opcode == RA_RS_GET_LITE_SUPPORT) {
        return -EPROTONOSUPPORT;
    }

    return 0;
}

extern int DlHalGetDeviceInfo(uint32_t devId, int32_t moduleType, int32_t infoType, int64_t* value);
int StubDlHalGetDeviceInfo(uint32_t devId, int32_t moduleType, int32_t infoType, int64_t* value)
{
    *value = (1 << 8);
    return 0;
}

void TcHdcRdevInit()
{
    mocker_clean();
    TcHdcTestEnvInit();
    struct rdev rdevInfo = {0};
    unsigned int rdevIndex = 0;
    struct RaRdmaHandle rdmaHandle = {0};
    mocker_invoke((stub_fn_t)RaHdcGetOpcodeLiteSupport, (stub_fn_t)StubRaHdcGetOpcodeLiteSupport, 100);
    mocker_invoke((stub_fn_t)RaHdcProcessMsg, (stub_fn_t)StubRaHdcProcessRdevInit, 100);
    int ret = RaHdcRdevInit(&rdmaHandle, NOTIFY, rdevInfo, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);
    RaHdcRdevDeinit(&rdmaHandle, NOTIFY);
    TcHdcTestEnvDeinit();

    mocker_clean();
    mocker((stub_fn_t)RaHdcNotifyBaseAddrInit, 1, 0);
    mocker((stub_fn_t)memcpy_s, 10, -10);
    ret = RaHdcRdevInit(&rdmaHandle, NOTIFY, rdevInfo, &rdevIndex);
    EXPECT_INT_EQ(ret, -ESAFEFUNC);

    mocker_clean();
    mocker((stub_fn_t)RaHdcNotifyBaseAddrInit, 1, 0);
    ret = RaHdcRdevInit(&rdmaHandle, NOTIFY, rdevInfo, &rdevIndex);
    EXPECT_INT_NE(ret, 0);

    mocker_clean();
    mocker((stub_fn_t)RaHdcNotifyBaseAddrInit, 1, 0);
    mocker((stub_fn_t)drvHdcAllocMsg, 1, -10);
    ret = RaHdcRdevInit(&rdmaHandle, NOTIFY, rdevInfo, &rdevIndex);
    EXPECT_INT_EQ(ret, -EINVAL);

    mocker_clean();
    mocker((stub_fn_t)RaHdcNotifyBaseAddrInit, 1, 0);
    mocker((stub_fn_t)RaHdcProcessMsg, 1, -1);
    mocker((stub_fn_t)halMemFree, 1, -1);
    ret = RaHdcRdevInit(&rdmaHandle, NOTIFY, rdevInfo, &rdevIndex);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker_invoke((stub_fn_t)RaHdcGetOpcodeLiteSupport, (stub_fn_t)StubRaHdcGetOpcodeLiteSupport, 100);
    mocker_invoke((stub_fn_t)RaHdcProcessMsg, (stub_fn_t)StubRaHdcProcessRdevInitError, 10);
    ret = RaHdcRdevInit(&rdmaHandle, NOTIFY, rdevInfo, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);
    RaHdcRdevDeinit(&rdmaHandle, NOTIFY);
    mocker_clean();

    mocker_invoke((stub_fn_t)RaHdcGetOpcodeLiteSupport, (stub_fn_t)StubRaHdcGetOpcodeLiteSupport, 100);
    mocker_invoke((stub_fn_t)RaHdcProcessMsg, (stub_fn_t)StubRaHdcProcessRdevInit, 100);
    mocker((stub_fn_t)RaRdmaLiteAllocCtx, 1, 0);
    ret = RaHdcRdevInit(&rdmaHandle, NOTIFY, rdevInfo, &rdevIndex);
    EXPECT_INT_EQ(ret, -EFAULT);
    mocker_clean();

    mocker_invoke((stub_fn_t)RaHdcGetOpcodeLiteSupport, (stub_fn_t)StubRaHdcGetOpcodeLiteSupport, 100);
    mocker_invoke((stub_fn_t)RaHdcProcessMsg, (stub_fn_t)StubRaHdcProcessRdevInit, 100);
    mocker((stub_fn_t)pthread_mutex_init, 1, -1);
    ret = RaHdcRdevInit(&rdmaHandle, NOTIFY, rdevInfo, &rdevIndex);
    EXPECT_INT_EQ(ret, -ESYSFUNC);
    mocker_clean();

    mocker_invoke((stub_fn_t)RaHdcGetOpcodeLiteSupport, (stub_fn_t)StubRaHdcGetOpcodeLiteSupport, 100);
    mocker_invoke((stub_fn_t)RaHdcProcessMsg, (stub_fn_t)StubRaHdcProcessRdevInit, 100);
    mocker((stub_fn_t)pthread_create, 1, -1);
    ret = RaHdcRdevInit(&rdmaHandle, NOTIFY, rdevInfo, &rdevIndex);
    EXPECT_INT_EQ(ret, -ESYSFUNC);
    mocker_clean();

    mocker_invoke((stub_fn_t)RaHdcGetOpcodeLiteSupport, (stub_fn_t)StubRaHdcGetOpcodeLiteSupport, 100);
    mocker_invoke((stub_fn_t)RaHdcProcessMsg, (stub_fn_t)StubRaHdcProcessRdevInit, 100);
    mocker((stub_fn_t)RaHdcGetLiteSupport, 1, -1);
    ret = RaHdcRdevInit(&rdmaHandle, NOTIFY, rdevInfo, &rdevIndex);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    unsigned int support = 0;
    mocker((stub_fn_t)DlDrvDeviceGetIndexByPhyId, 1, 0);
    mocker((stub_fn_t)DlHalMemCtl, 1, 0);
    RaHdcGetDrvLiteSupport(0, true, &support);
    EXPECT_INT_EQ(support, 0);
    mocker_clean();

    mocker((stub_fn_t)DlDrvDeviceGetIndexByPhyId, 1, 0);
    mocker_invoke((stub_fn_t)DlHalGetDeviceInfo, (stub_fn_t)StubDlHalGetDeviceInfo, 10);
    mocker((stub_fn_t)DlHalMemCtl, 1, 0);
    RaHdcGetDrvLiteSupport(0, false, &support);
    EXPECT_INT_EQ(support, 0);
    mocker_clean();
}

void TcHdcRdevDeinit()
{
    mocker_clean();
    TcHdcTestEnvInit();
    struct RaRdmaHandle rdmaHandle = {0};
    int ret = RaHdcRdevDeinit(&rdmaHandle, NOTIFY);
    EXPECT_INT_EQ(ret, 0);
    TcHdcTestEnvDeinit();

    mocker((stub_fn_t)RaHdcProcessMsg, 1, 0);
    mocker((stub_fn_t)RaHdcNotifyCfgGet, 1, -1);
    ret = RaHdcRdevDeinit(&rdmaHandle, NOTIFY);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker((stub_fn_t)RaHdcProcessMsg, 1, 0);
    mocker((stub_fn_t)RaHdcNotifyCfgGet, 1, 0);
    mocker((stub_fn_t)halMemFree, 1, 2);
    ret = RaHdcRdevDeinit(&rdmaHandle, NOTIFY);
    EXPECT_INT_EQ(ret, -2);
    mocker_clean();
}

void TcHdcSocketWhiteListAddV2()
{
    mocker_clean();
    TcHdcTestEnvInit();
    struct rdev rdevInfo = {0};
    struct SocketWlistInfoT whiteList[1] = {{0}};
    gInterfaceVersion = RA_RS_WLIST_ADD_V2_VERSION;
    mocker_invoke((stub_fn_t)RaGetInterfaceVersion, (stub_fn_t)StubRaGetInterfaceVersion, 1);
    int ret = RaHdcSocketWhiteListAdd(rdevInfo, whiteList, 1);
    gInterfaceVersion = 0;
    EXPECT_INT_EQ(ret, 0);
    TcHdcTestEnvDeinit();
}

void TcHdcSocketWhiteListAdd()
{
    mocker_clean();
    TcHdcTestEnvInit();
    struct rdev rdevInfo = {0};
    struct SocketWlistInfoT whiteList[1] = {{0}};
    int ret = RaHdcSocketWhiteListAdd(rdevInfo, whiteList, 1);
    EXPECT_INT_EQ(ret, 0);
    TcHdcTestEnvDeinit();

    TcHdcSocketWhiteListAddV2();
}

void TcHdcSocketWhiteListDelV2()
{
    mocker_clean();
    TcHdcTestEnvInit();
    struct rdev rdevInfo = {0};
    struct SocketWlistInfoT whiteList[1] = {{0}};
    int ret = RaHdcSocketWhiteListDel(rdevInfo, whiteList, 1);
    EXPECT_INT_EQ(ret, 0);
    TcHdcTestEnvDeinit();

    mocker_clean();
    mocker((stub_fn_t)calloc, 1, NULL);
    gInterfaceVersion = RA_RS_WLIST_DEL_V2_VERSION;
    mocker_invoke((stub_fn_t)RaGetInterfaceVersion, (stub_fn_t)StubRaGetInterfaceVersion, 1);
    ret = RaHdcSocketWhiteListDel(rdevInfo, whiteList, 1);
    gInterfaceVersion = 0;
    EXPECT_INT_EQ(ret, -ENOMEM);

    mocker_clean();
    mocker((stub_fn_t)memcpy_s, 1, -1);
    gInterfaceVersion = RA_RS_WLIST_DEL_V2_VERSION;
    mocker_invoke((stub_fn_t)RaGetInterfaceVersion, (stub_fn_t)StubRaGetInterfaceVersion, 1);
    ret = RaHdcSocketWhiteListDel(rdevInfo, whiteList, 1);
    gInterfaceVersion = 0;
    EXPECT_INT_EQ(ret, -ESAFEFUNC);

    mocker_clean();
    mocker((stub_fn_t)drvHdcAllocMsg, 1, -10);
    gInterfaceVersion = RA_RS_WLIST_DEL_V2_VERSION;
    mocker_invoke((stub_fn_t)RaGetInterfaceVersion, (stub_fn_t)StubRaGetInterfaceVersion, 1);
    ret = RaHdcSocketWhiteListDel(rdevInfo, whiteList, 1);
    gInterfaceVersion = 0;
    EXPECT_INT_EQ(ret, -EINVAL);
    mocker_clean();
}

void TcHdcSocketWhiteListDel()
{
    mocker_clean();
    TcHdcTestEnvInit();
    struct rdev rdevInfo = {0};
    struct SocketWlistInfoT whiteList[1] = {{0}};
    int ret = RaHdcSocketWhiteListDel(rdevInfo, whiteList, 1);
    EXPECT_INT_EQ(ret, 0);
    TcHdcTestEnvDeinit();

    mocker_clean();
    mocker((stub_fn_t)calloc, 1, NULL);
    ret = RaHdcSocketWhiteListDel(rdevInfo, whiteList, 1);
    EXPECT_INT_EQ(ret, -ENOMEM);

    mocker_clean();
    mocker((stub_fn_t)memcpy_s, 1, -1);
    ret = RaHdcSocketWhiteListDel(rdevInfo, whiteList, 1);
    EXPECT_INT_EQ(ret, -ESAFEFUNC);

    mocker_clean();
    mocker((stub_fn_t)drvHdcAllocMsg, 1, -10);
    ret = RaHdcSocketWhiteListDel(rdevInfo, whiteList, 1);
    EXPECT_INT_EQ(ret, -EINVAL);
    mocker_clean();

    TcHdcSocketWhiteListDelV2();
}

void TcHdcGetIfaddrs()
{
    mocker_clean();
    TcHdcTestEnvInit();
    struct IfaddrInfo infos[1] = {{0}};
    unsigned int num = 1;
    int ret = RaHdcGetIfaddrs(0, infos, &num);
    EXPECT_INT_EQ(ret, 0);
    TcHdcTestEnvDeinit();
    mocker(drvDeviceGetPhyIdByIndex, 1, -1);
    ret = RaHdcGetIfaddrs(0, infos, &num);
    EXPECT_INT_EQ(ret, -EINVAL);
}

void TcHdcGetIfaddrsV2()
{
    mocker_clean();
    TcHdcTestEnvInit();
    struct InterfaceInfo infos[1] = {{0}};
    unsigned int num = 1;
    int ret = RaHdcGetIfaddrsV2(0, 0, infos, &num);
    EXPECT_INT_EQ(ret, 0);
    TcHdcTestEnvDeinit();
    mocker(drvDeviceGetPhyIdByIndex, 1, -1);
    ret = RaHdcGetIfaddrsV2(0, 0, infos, &num);
    EXPECT_INT_EQ(ret, -EINVAL);
}

void TcHdcGetIfnum()
{
    mocker_clean();
    TcHdcTestEnvInit();

    unsigned int num = 1;
    int ret = RaHdcGetIfnum(0, 0, &num);
    EXPECT_INT_EQ(ret, 0);
    TcHdcTestEnvDeinit();

    mocker(drvDeviceGetPhyIdByIndex, 1, -1);
    ret = RaHdcGetIfnum(0, 0, &num);
    EXPECT_INT_EQ(ret, -EINVAL);
}

void TcHdcMessageProcessFail()
{
    mocker_clean();
    struct ProcessRaSign pRaSign;
    pRaSign.tgid = 0;
    struct RaInitConfig config = {
        .phyId = gDevid,
        .nicPosition = NETWORK_OFFLINE,
        .hdcType = HDC_SERVICE_TYPE_RDMA,
    };
    mocker((stub_fn_t)drvHdcClientCreate, 1, 0);
    mocker_invoke((stub_fn_t)drvHdcSessionConnect, (stub_fn_t)StubSessionConnect, 1);
    mocker((stub_fn_t)drvHdcSetSessionReference, 1, 0);
    mocker((stub_fn_t)halHdcRecv, 2, 0);
    int ret = RaHdcInit(&config, pRaSign);
    EXPECT_INT_EQ(ret, 0);

    struct IfaddrInfo infos[1] = {{0}};
    unsigned int num = 1;
    mocker_clean();
    mocker((stub_fn_t)calloc, 1, NULL);
    ret = RaHdcGetIfaddrs(0, infos, &num);
    EXPECT_INT_EQ(ret, -ENOMEM);

    mocker_clean();
    mocker((stub_fn_t)memcpy_s, 1, -1);
    ret = RaHdcGetIfaddrs(0, infos, &num);
    EXPECT_INT_EQ(ret, -ESAFEFUNC);

    mocker_clean();
    mocker((stub_fn_t)drvHdcAllocMsg, 1, 10);
    ret = RaHdcGetIfaddrs(0, infos, &num);
    EXPECT_INT_EQ(ret, -10);

    mocker_clean();
    mocker((stub_fn_t)drvHdcAddMsgBuffer, 1, 10);
    ret = RaHdcGetIfaddrs(0, infos, &num);
    EXPECT_INT_EQ(ret, -10);

    mocker_clean();
    mocker((stub_fn_t)halHdcSend, 1, 10);
    ret = RaHdcGetIfaddrs(0, infos, &num);
    EXPECT_INT_EQ(ret, -10);

    mocker_clean();
    mocker((stub_fn_t)drvHdcReuseMsg, 1, 10);
    ret = RaHdcGetIfaddrs(0, infos, &num);
    EXPECT_INT_EQ(ret, -10);

    mocker_clean();
    mocker((stub_fn_t)halHdcRecv, 1, 10);
    ret = RaHdcGetIfaddrs(0, infos, &num);
    EXPECT_INT_EQ(ret, -10);

    mocker_clean();
    mocker((stub_fn_t)halHdcRecv, 1, 0);
    mocker((stub_fn_t)drvHdcGetMsgBuffer, 1, 10);
    ret = RaHdcGetIfaddrs(0, infos, &num);
    EXPECT_INT_EQ(ret, -10);

    mocker_clean();
    mocker((stub_fn_t)halHdcRecv, 1, 0);
    TcHdcTestEnvDeinit();
}

void TcHdcSocketRecvFail()
{
    struct SocketHdcInfo socketInfo = {0};
    char sendBuf[16] = {0};

    mocker_clean();
    TcHdcTestEnvInit();

    mocker_invoke((stub_fn_t)drvHdcGetMsgBuffer, (stub_fn_t)stub_drvHdcGetMsgBuffer, 2);
    struct MsgHead* testHead = (struct MsgHead*)gDrvRecvMsg;
    testHead->ret = -EAGAIN;
    testHead->opcode = RA_RS_SOCKET_RECV;
    gDrvRecvMsgLen = sizeof(union OpSocketRecvData) + sizeof(sendBuf) + sizeof(struct MsgHead);
    int ret = RaHdcSocketRecv(gDevid, &socketInfo, sendBuf, sizeof(sendBuf));
    EXPECT_INT_EQ(ret, -EAGAIN);

    testHead->ret = -100;
    ret = RaHdcSocketRecv(gDevid, &socketInfo, sendBuf, sizeof(sendBuf));
    EXPECT_INT_EQ(ret, -100);

    mocker((stub_fn_t)drvHdcAddMsgBuffer, 10, 10);
    ret = RaHdcSocketRecv(gDevid, &socketInfo, sendBuf, sizeof(sendBuf));
    EXPECT_INT_EQ(ret, -10);
    mocker((stub_fn_t)RaHdcSessionClose, 10, 0);
    TcHdcTestEnvDeinit();
}

extern void RaHdcSendWrlistExtInitV2(
    union OpSendWrlistDataExtV2* sendWrlist, struct RaQpHandle* qpHdc, unsigned int completeCnt,
    struct WrlistSendCompleteNum wrlistNum);
extern void RaHdcSendWrlistExtInit(
    union OpSendWrlistDataExt* sendWrlist, struct RaQpHandle* qpHdc, unsigned int completeCnt,
    struct WrlistSendCompleteNum wrlistNum);
void TcRaHdcSendWrlistExtInitV2()
{
    union OpSendWrlistDataExtV2 sendWrlist;
    struct RaQpHandle qpHdc = {0};
    unsigned int completeCnt;
    struct WrlistSendCompleteNum wrlistNum = {0};
    RaHdcSendWrlistExtInitV2(&sendWrlist, &qpHdc, completeCnt, wrlistNum);
}

void TcRaHdcSendWrlistExtInit()
{
    union OpSendWrlistDataExt sendWrlist;
    struct RaQpHandle qpHdc = {0};
    unsigned int completeCnt;
    struct WrlistSendCompleteNum wrlistNum = {0};
    RaHdcSendWrlistExtInit(&sendWrlist, &qpHdc, completeCnt, wrlistNum);

    TcRaHdcSendWrlistExtInitV2();
}

void TcRaHdcSendWrlistExtV2()
{
    mocker_clean();

    struct RaQpHandle qpHandle = {0};
    int ret;
    struct SendWrlistDataExt wr[1] = {{0}};
    struct SendWrRsp opRsp[1] = {{0}};
    struct WrlistSendCompleteNum wrlistNum = {0};
    unsigned int completeNum = 0;
    wrlistNum.sendNum = 1;
    wrlistNum.completeNum = &completeNum;

    struct RaQpHandle handle = {0};
    handle.qpMode = 1;

    mocker_invoke((stub_fn_t)RaHdcProcessMsg, (stub_fn_t)StubRaHdcProcessMsgWrlist, 1);
    gInterfaceVersion = RA_RS_SEND_WRLIST_EXT_V2_VERSION;
    mocker_invoke((stub_fn_t)RaGetInterfaceVersion, (stub_fn_t)StubRaGetInterfaceVersion, 1);
    RaHdcSendWrlistExt(&qpHandle, wr, opRsp, wrlistNum);
    gInterfaceVersion = 0;
    mocker_clean();

    mocker_invoke((stub_fn_t)RaHdcProcessMsg, (stub_fn_t)StubRaHdcProcessMsgWrlist3, 1);
    gInterfaceVersion = RA_RS_SEND_WRLIST_EXT_V2_VERSION;
    mocker_invoke((stub_fn_t)RaGetInterfaceVersion, (stub_fn_t)StubRaGetInterfaceVersion, 1);
    ret = RaHdcSendWrlistExt(&qpHandle, wr, opRsp, wrlistNum);
    gInterfaceVersion = 0;
    EXPECT_INT_EQ(ret, -EINVAL);
    mocker_clean();

    mocker((stub_fn_t)memcpy_s, 1, -1);
    gInterfaceVersion = RA_RS_SEND_WRLIST_EXT_V2_VERSION;
    mocker_invoke((stub_fn_t)RaGetInterfaceVersion, (stub_fn_t)StubRaGetInterfaceVersion, 1);
    ret = RaHdcSendWrlistExt(&qpHandle, wr, opRsp, wrlistNum);
    gInterfaceVersion = 0;
    EXPECT_INT_EQ(ret, -ESAFEFUNC);
    mocker_clean();
}

void TcRaHdcSendWrlistExt()
{
    mocker_clean();

    struct RaQpHandle qpHandle = {0};
    int ret;
    struct SendWrlistDataExt wr[1] = {{0}};
    struct SendWrRsp opRsp[1] = {{0}};
    struct WrlistSendCompleteNum wrlistNum = {0};
    unsigned int completeNum = 0;
    wrlistNum.sendNum = 1;
    wrlistNum.completeNum = &completeNum;

    struct RaQpHandle handle = {0};
    handle.qpMode = 1;

    mocker_invoke((stub_fn_t)RaHdcProcessMsg, (stub_fn_t)StubRaHdcProcessMsgWrlist, 1);
    RaHdcSendWrlistExt(&qpHandle, wr, opRsp, wrlistNum);
    mocker_clean();

    mocker_invoke((stub_fn_t)RaHdcProcessMsg, (stub_fn_t)StubRaHdcProcessMsgWrlist3, 1);
    ret = RaHdcSendWrlistExt(&qpHandle, wr, opRsp, wrlistNum);
    EXPECT_INT_EQ(ret, -EINVAL);
    mocker_clean();

    TcRaHdcSendWrlistExtV2();
}

void TcRaHdcSendNormalWrlist()
{
    struct RaQpHandle qpHandle = {0};
    struct WrInfo wr[1] = {{0}};
    struct SendWrRsp opRsp[1] = {{0}};
    struct WrlistSendCompleteNum wrlistNum = {0};
    unsigned int completeNum = 0;
    wrlistNum.sendNum = 1;
    wrlistNum.completeNum = &completeNum;
    int ret = 0;

    mocker_clean();

    qpHandle.qpMode = RA_RS_OP_QP_MODE;
    qpHandle.supportLite = 1;
    RaHdcSendNormalWrlist(&qpHandle, wr, opRsp, wrlistNum);

    qpHandle.qpMode = RA_RS_NOR_QP_MODE;
    wrlistNum.completeNum = &completeNum;
    mocker_invoke((stub_fn_t)RaHdcProcessMsg, (stub_fn_t)StubRaHdcProcessMsgWrlist, 1);
    ret = RaHdcSendNormalWrlist(&qpHandle, wr, opRsp, wrlistNum);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRaHdcSetQpAttrQos()
{
    struct QosAttr attr = {0};
    attr.tc = 33 * 4;
    attr.sl = 4;

    mocker_clean();
    TcHdcTestEnvInit();
    struct RaRdmaHandle rdmaHandle = {0};
    void* qpHandle = NULL;
    int ret = RaHdcQpCreate(&rdmaHandle, 0, 0, &qpHandle);
    char qpReg[16] = {0};
    ASSERT_ADDR_NE(qpHandle, NULL);

    ret = RaHdcSetQpAttrQos(qpHandle, &attr);
    EXPECT_INT_EQ(ret, 0);
    ret = RaHdcQpDestroy(qpHandle);
    EXPECT_INT_EQ(ret, 0);
    TcHdcTestEnvDeinit();

    mocker_clean();
    mocker((stub_fn_t)calloc, 1, NULL);
    struct RaQpHandle testQpHandle = {0};
    ret = RaHdcSetQpAttrQos(&testQpHandle, &attr);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();
}

void TcRaHdcSetQpAttrTimeout()
{
    unsigned int timeout = 15;

    mocker_clean();
    TcHdcTestEnvInit();
    struct RaRdmaHandle rdmaHandle = {0};
    void* qpHandle = NULL;
    int ret = RaHdcQpCreate(&rdmaHandle, 0, 0, &qpHandle);
    char qpReg[16] = {0};
    ASSERT_ADDR_NE(qpHandle, NULL);

    ret = RaHdcSetQpAttrTimeout(qpHandle, &timeout);
    EXPECT_INT_EQ(ret, 0);
    ret = RaHdcQpDestroy(qpHandle);
    EXPECT_INT_EQ(ret, 0);
    TcHdcTestEnvDeinit();

    mocker_clean();
    mocker((stub_fn_t)calloc, 1, NULL);
    struct RaQpHandle testQpHandle = {0};
    ret = RaHdcSetQpAttrTimeout(&testQpHandle, &timeout);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();
}

void TcRaHdcSetQpAttrRetryCnt()
{
    unsigned int retryCnt = 6;

    mocker_clean();
    TcHdcTestEnvInit();
    struct RaRdmaHandle rdmaHandle = {0};
    void* qpHandle = NULL;
    int ret = RaHdcQpCreate(&rdmaHandle, 0, 0, &qpHandle);
    char qpReg[16] = {0};
    ASSERT_ADDR_NE(qpHandle, NULL);

    ret = RaHdcSetQpAttrRetryCnt(qpHandle, &retryCnt);
    EXPECT_INT_EQ(ret, 0);
    ret = RaHdcQpDestroy(qpHandle);
    EXPECT_INT_EQ(ret, 0);
    TcHdcTestEnvDeinit();

    mocker_clean();
    mocker((stub_fn_t)calloc, 1, NULL);
    struct RaQpHandle testQpHandle = {0};
    ret = RaHdcSetQpAttrRetryCnt(&testQpHandle, &retryCnt);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();
}

void TcRaHdcGetCqeErrInfo()
{
    mocker_clean();
    TcHdcTestEnvInit();
    int ret;
    struct CqeErrInfo info = {0};
    ret = RaHdcGetCqeErrInfo(0, &info);
    EXPECT_INT_EQ(ret, 0);

    struct RaQpHandle qpHdc = {0};
    qpHdc.phyId = 0;
    RaHdcLiteSaveCqeErrInfo(&qpHdc, 0x12);
    ret = RaHdcGetCqeErrInfo(0, &info);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(info.status, 0x12);
    RaHdcLiteSaveCqeErrInfo(&qpHdc, 0x15);
    ret = RaHdcGetCqeErrInfo(0, &info);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(info.status, 0x15);
    TcHdcTestEnvDeinit();

    ret = RaHdcGetCqeErrInfo(0, &info);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

int StubRaHdcGetCqeErrNum(unsigned int opcode, int deviceId, char* data, unsigned int dataSize)
{
    if (dataSize == sizeof(union OpGetCqeErrInfoNumData)) {
        union OpGetCqeErrInfoNumData* cqeErrInfoNumData = (union OpGetCqeErrInfoNumData*)data;
        cqeErrInfoNumData->rxData.num = 10;
    } else if (dataSize == sizeof(union OpGetCqeErrInfoListData)) {
        union OpGetCqeErrInfoListData* cqeErrInfoList = (union OpGetCqeErrInfoListData*)data;
        cqeErrInfoList->rxData.num = 1;
    }
    return 0;
}

int StubRaHdcGetCqeErrNumV2(unsigned int opcode, int deviceId, char* data, unsigned int dataSize)
{
    StubRaHdcGetCqeErrNum(opcode, deviceId, data, dataSize);
    ((union OpGetCqeErrInfoListData*)data)->rxData.infoList[0].qpn = 12345;
    return 0;
}

int StubRaHdcGetCqeErrInfoNum(struct RaRdmaHandle* rdmaHandle, unsigned int* num)
{
    *num = 10;
    return 0;
}

int StubRaHdcGetInterfaceVersion(unsigned int phyId, unsigned int interfaceOpcode, unsigned int* interfaceVersion)
{
    *interfaceVersion = 1;
    return 0;
}

int StubRaHdcLiteGetCqeErrInfoList(struct RaRdmaHandle* rdmaHandle, struct CqeErrInfo* infoList, unsigned int* num)
{
    *num = 10;
    return 0;
}

void TcRaHdcGetCqeErrInfoList()
{
    struct RaRdmaHandle rdmaHandle = {0};
    struct RaQpHandle qpHdc = {0};
    struct CqeErrInfo info[130] = {0};
    int num = 11;
    int ret = 0;

    mocker_clean();
    mocker_invoke(RaHdcLiteGetCqeErrInfoList, StubRaHdcLiteGetCqeErrInfoList, 1);
    mocker_invoke(RaHdcGetCqeErrInfoNum, StubRaHdcGetCqeErrInfoNum, 1);
    mocker((stub_fn_t)RaHdcProcessMsg, 10, 0);
    ret = RaHdcGetCqeErrInfoList(&rdmaHandle, info, &num);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(num, 10);

    mocker_clean();
    mocker_invoke(RaHdcLiteGetCqeErrInfoList, StubRaHdcLiteGetCqeErrInfoList, 1);
    mocker_invoke(RaHdcGetCqeErrInfoNum, StubRaHdcGetCqeErrInfoNum, 1);
    mocker_invoke((stub_fn_t)RaHdcProcessMsg, (stub_fn_t)StubRaHdcGetCqeErrNum, 10);
    num = 11;
    ret = RaHdcGetCqeErrInfoList(&rdmaHandle, info, &num);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(num, 11);

    mocker_clean();
    mocker_invoke(RaHdcLiteGetCqeErrInfoList, StubRaHdcLiteGetCqeErrInfoList, 1);
    mocker_invoke(RaHdcGetCqeErrInfoNum, StubRaHdcGetCqeErrInfoNum, 1);
    mocker_invoke((stub_fn_t)RaHdcProcessMsg, (stub_fn_t)StubRaHdcGetCqeErrNumV2, 10);
    num = 129;
    ret = RaHdcGetCqeErrInfoList(&rdmaHandle, info, &num);
    EXPECT_INT_EQ(info[10].qpn, 12345);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(num, 11);

    mocker_clean();
    mocker_invoke(RaHdcLiteGetCqeErrInfoList, StubRaHdcLiteGetCqeErrInfoList, 1);
    mocker_invoke(RaHdcGetCqeErrInfoNum, StubRaHdcGetCqeErrInfoNum, 1);
    mocker_invoke((stub_fn_t)RaHdcProcessMsg, (stub_fn_t)StubRaHdcGetCqeErrNum, 10);
    mocker((stub_fn_t)memcpy_s, 1, -1);
    num = 11;
    ret = RaHdcGetCqeErrInfoList(&rdmaHandle, info, &num);
    EXPECT_INT_EQ(ret, -1);

    mocker_clean();
    rdmaHandle.supportLite = 0;
    ret = RaHdcLiteGetCqeErrInfoList(&rdmaHandle, info, &num);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(num, 0);

    mocker_clean();
    rdmaHandle.supportLite = 1;
    rdmaHandle.cqeErrCnt = 0;
    mocker(pthread_mutex_lock, 10, 0);
    mocker(pthread_mutex_unlock, 10, 0);
    ret = RaHdcLiteGetCqeErrInfoList(&rdmaHandle, info, &num);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(num, 0);

    mocker_clean();
    mocker(RaHdcGetInterfaceVersion, 10, -1);
    ret = RaHdcGetCqeErrInfoNum(&rdmaHandle, &num);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(num, 0);

    mocker_clean();
    mocker(RaHdcProcessMsg, 10, 0);
    mocker_invoke(RaHdcGetInterfaceVersion, StubRaHdcGetInterfaceVersion, 1);
    ret = RaHdcGetCqeErrInfoNum(&rdmaHandle, &num);
    EXPECT_INT_EQ(ret, 0);
    return;
}

void TcRaHdcQpCreateOp()
{
    mocker_clean();
    TcHdcTestEnvInit();
    struct RaRdmaHandle rdmaHandle = {0};
    void* qpHandle = NULL;
    rdmaHandle.supportLite = 1;
    RA_INIT_LIST_HEAD(&rdmaHandle.qpList);
    int ret = RaHdcQpCreate(&rdmaHandle, 0, 2, (void**)&qpHandle);
    EXPECT_INT_EQ(ret, 0);
    ASSERT_ADDR_NE(qpHandle, NULL);
    struct rdma_lite_qp_cap cap;

    ret = RaHdcQpDestroy(qpHandle);
    EXPECT_INT_EQ(ret, 0);
    TcHdcTestEnvDeinit();

    mocker_clean();

    mocker((stub_fn_t)RaHdcProcessMsg, 10, 0);
    mocker((stub_fn_t)RaRdmaLiteCreateCq, 1, 0);
    ret = RaHdcQpCreate(&rdmaHandle, 0, 2, (void**)&qpHandle);
    EXPECT_INT_EQ(ret, -EFAULT);
    mocker_clean();

    mocker((stub_fn_t)RaHdcProcessMsg, 10, 0);
    mocker((stub_fn_t)RaRdmaLiteCreateQp, 1, 0);
    ret = RaHdcQpCreate(&rdmaHandle, 0, 2, (void**)&qpHandle);
    EXPECT_INT_EQ(ret, -EFAULT);
    mocker_clean();

    mocker((stub_fn_t)RaHdcProcessMsg, 10, 0);
    mocker((stub_fn_t)pthread_mutex_init, 1, -1);
    ret = RaHdcQpCreate(&rdmaHandle, 0, 2, (void**)&qpHandle);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    struct RaQpHandle qpHdc = {0};
    qpHdc.supportLite = 1;
    qpHdc.qpMode = 2;
    cap.max_inline_data = QP_DEFAULT_MAX_CAP_INLINE_DATA;
    cap.max_send_sge = QP_DEFAULT_MIN_CAP_SEND_SGE;
    cap.max_recv_sge = 1;
    cap.max_send_wr = RA_QP_TX_DEPTH;
    cap.max_recv_wr = RA_QP_TX_DEPTH;
    mocker((stub_fn_t)RaHdcProcessMsg, 10, -1);
    ret = RaHdcLiteQpCreate(&rdmaHandle, &qpHdc, &cap);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRaHdcGetQpStatusOp()
{
    int status;

    mocker_clean();
    TcHdcTestEnvInit();
    struct RaRdmaHandle rdmaHandle = {0};
    void* qpHandle = NULL;
    rdmaHandle.supportLite = 1;
    RA_INIT_LIST_HEAD(&rdmaHandle.qpList);
    int ret = RaHdcQpCreate(&rdmaHandle, 0, 2, (void**)&qpHandle);
    EXPECT_INT_EQ(ret, 0);
    ASSERT_ADDR_NE(qpHandle, NULL);

    status = 0;
    mocker((stub_fn_t)RaHdcGetInterfaceVersion, 10, -22);
    ret = RaHdcGetQpStatus(qpHandle, &status);
    EXPECT_INT_EQ(ret, 0);

    gInterfaceVersion = RA_RS_OPCODE_BASE_VERSION;
    status = 0;
    mocker_invoke(RaHdcGetInterfaceVersion, StubRaGetInterfaceVersion, 10);
    ret = RaHdcGetQpStatus(qpHandle, &status);
    EXPECT_INT_EQ(ret, 0);

    ret = RaHdcQpDestroy(qpHandle);
    EXPECT_INT_EQ(ret, 0);
    TcHdcTestEnvDeinit();

    mocker_clean();

    struct RaQpHandle qpHdc = {0};
    qpHdc.supportLite = 1;
    qpHdc.qpMode = 2;
    mocker((stub_fn_t)RaHdcProcessMsg, 10, -1);
    ret = RaHdcLiteGetConnectedInfo(&qpHdc);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker((stub_fn_t)RaHdcProcessMsg, 10, 0);
    mocker((stub_fn_t)RaRdmaLiteSetQpSl, 1, -1);
    ret = RaHdcLiteGetConnectedInfo(&qpHdc);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

extern int RaRdmaLitePollCq(struct rdma_lite_cq* liteCq, int numEntries, struct rdma_lite_wc* liteWc);
int stub_RaRdmaLitePollCq(struct rdma_lite_cq* liteCq, int numEntries, struct rdma_lite_wc* liteWc)
{
    int i = 0;
    for (i = 0; i < numEntries; i++) {
        liteWc[i].status = 0x12;
    }

    return 2;
}

void TcHdcSendWrOp()
{
    mocker_clean();
    TcHdcTestEnvInit();
    struct rdev rdevInfo = {0};
    unsigned int rdevIndex = 0;
    struct RaRdmaHandle rdmaHandle = {0};
    struct RaQpHandle* qpHandle = NULL;

    mocker_invoke((stub_fn_t)RaHdcGetOpcodeLiteSupport, (stub_fn_t)StubRaHdcGetOpcodeLiteSupport, 100);
    mocker_invoke((stub_fn_t)RaHdcProcessMsg, (stub_fn_t)StubRaHdcProcessRdevInit, 100);
    int ret = RaHdcRdevInit(&rdmaHandle, NOTIFY, rdevInfo, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    ret = RaHdcQpCreate(&rdmaHandle, 0, 2, (void**)&qpHandle);
    EXPECT_INT_EQ(ret, 0);

    struct SendWr wr = {0};
    struct SendWrRsp rsp = {0};
    int i = 0;

    ASSERT_ADDR_NE(qpHandle, NULL);

    void* addr = malloc(10);
    struct SgList mem;
    mem.addr = (uint64_t)(uintptr_t)addr;
    mem.len = 10;
    wr.bufList = &mem;
    wr.dstAddr = 0x111;
    wr.bufNum = 1;
    wr.op = 0;
    wr.sendFlag = 0;
    qpHandle->localMr[0].addr = wr.bufList[0].addr;
    qpHandle->localMr[0].len = 0x10000;
    qpHandle->remMr[0].addr = wr.dstAddr;
    qpHandle->remMr[0].len = 0x10000;
    qpHandle->sendWrNum = 999;
    mocker_invoke((stub_fn_t)RaRdmaLitePollCq, stub_RaRdmaLitePollCq, 10);
    ret = RaHdcSendWr(qpHandle, &wr, &rsp);
    EXPECT_INT_EQ(ret, 0);

    unsigned int completeNum = 0;
    struct WrlistSendCompleteNum wrlistNum = {0};
    wrlistNum.sendNum = 2;
    wrlistNum.completeNum = &completeNum;
    struct SendWrlistData wrlist[wrlistNum.sendNum];
    struct SendWrRsp opRspList[wrlistNum.sendNum];
    wrlist[0].memList = mem;
    wrlist[0].dstAddr = 0x111;
    wrlist[0].op = 0;
    wrlist[0].sendFlags = 0;
    wrlist[1].memList = mem;
    wrlist[1].dstAddr = 0x111;
    wrlist[1].op = 0;
    wrlist[1].sendFlags = 0;
    qpHandle->rdmaOps = rdmaHandle.rdmaOps;
    gInterfaceVersion = RA_RS_SEND_WRLIST_V2_VERSION;
    mocker_invoke((stub_fn_t)RaGetInterfaceVersion, (stub_fn_t)StubRaGetInterfaceVersion, 1);
    ret = RaHdcSendWrlist(qpHandle, wrlist, opRspList, wrlistNum);
    gInterfaceVersion = 0;
    EXPECT_INT_EQ(0, ret);

    struct SendWrlistDataExt dataExt[wrlistNum.sendNum];
    dataExt[0].memList = mem;
    dataExt[0].dstAddr = 0x111;
    dataExt[0].op = 0;
    dataExt[0].sendFlags = 0;
    dataExt[1].memList = mem;
    dataExt[1].dstAddr = 0x111;
    dataExt[1].op = 0;
    dataExt[1].sendFlags = 0;
    gInterfaceVersion = RA_RS_SEND_WRLIST_EXT_V2_VERSION;
    mocker_invoke((stub_fn_t)RaGetInterfaceVersion, (stub_fn_t)StubRaGetInterfaceVersion, 1);
    ret = RaHdcSendWrlistExt(qpHandle, dataExt, opRspList, wrlistNum);
    gInterfaceVersion = 0;
    EXPECT_INT_EQ(0, ret);

    mocker((stub_fn_t)RaHdcLitePostSend, 10, -1);
    gInterfaceVersion = RA_RS_SEND_WRLIST_V2_VERSION;
    mocker_invoke((stub_fn_t)RaGetInterfaceVersion, (stub_fn_t)StubRaGetInterfaceVersion, 1);
    ret = RaHdcSendWrlist(qpHandle, wrlist, opRspList, wrlistNum);
    gInterfaceVersion = 0;
    EXPECT_INT_EQ(ret, -1);

    mocker((stub_fn_t)RaHdcLitePostSend, 10, -1);
    gInterfaceVersion = RA_RS_SEND_WRLIST_EXT_V2_VERSION;
    mocker_invoke((stub_fn_t)RaGetInterfaceVersion, (stub_fn_t)StubRaGetInterfaceVersion, 1);
    ret = RaHdcSendWrlistExt(qpHandle, dataExt, opRspList, wrlistNum);
    gInterfaceVersion = 0;
    EXPECT_INT_EQ(ret, -1);

    RaHdcLitePeriodPollCqe(&rdmaHandle);

    ret = RaHdcQpDestroy(qpHandle);
    EXPECT_INT_EQ(ret, 0);

    RaHdcRdevDeinit(&rdmaHandle, NOTIFY);

    TcHdcTestEnvDeinit();

    free(addr);
    mocker_clean();
}

void TcHdcLiteSendWrOp()
{
    mocker_clean();
    TcHdcTestEnvInit();
    struct rdev rdevInfo = {0};
    unsigned int rdevIndex = 0;
    struct RaRdmaHandle rdmaHandle = {0};
    struct RaQpHandle* qpHandle = NULL;

    mocker_invoke((stub_fn_t)RaHdcGetOpcodeLiteSupport, (stub_fn_t)StubRaHdcGetOpcodeLiteSupport, 100);
    mocker_invoke((stub_fn_t)RaHdcProcessMsg, (stub_fn_t)StubRaHdcProcessRdevInit, 100);
    int ret = RaHdcRdevInit(&rdmaHandle, NOTIFY, rdevInfo, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    ret = RaHdcQpCreate(&rdmaHandle, 0, 2, (void**)&qpHandle);
    EXPECT_INT_EQ(ret, 0);

    struct SendWr wr = {0};
    struct SendWrV2 wrV2 = {0};
    struct SendWrRsp rsp = {0};
    int i = 0;

    ASSERT_ADDR_NE(qpHandle, NULL);

    void* addr = malloc(10);
    struct SgList mem;
    mem.addr = (uint64_t)(uintptr_t)addr;
    mem.len = 10;
    wr.bufList = &mem;
    wr.dstAddr = 0x111;
    wr.bufNum = 1;
    wr.op = 0;
    wr.sendFlag = 0;
    qpHandle->localMr[0].addr = wr.bufList[0].addr;
    qpHandle->localMr[0].len = 0x10000;
    qpHandle->remMr[0].addr = wr.dstAddr;
    qpHandle->remMr[0].len = 0x10000;
    qpHandle->sendWrNum = 999;
    qpHandle->supportLite = 1;
    mocker_invoke((stub_fn_t)RaRdmaLitePollCq, stub_RaRdmaLitePollCq, 10);
    ret = RaHdcSendWr(qpHandle, &wr, &rsp);
    EXPECT_INT_EQ(ret, 0);

    unsigned int completeNum = 0;
    struct WrlistSendCompleteNum wrlistNum = {0};
    wrlistNum.sendNum = 2;
    wrlistNum.completeNum = &completeNum;
    struct SendWrlistData wrlist[wrlistNum.sendNum];
    struct SendWrRsp opRspList[wrlistNum.sendNum];
    wrlist[0].memList = mem;
    wrlist[0].dstAddr = 0x111;
    wrlist[0].op = 0;
    wrlist[0].sendFlags = 0;
    wrlist[1].memList = mem;
    wrlist[1].dstAddr = 0x111;
    wrlist[1].op = 0;
    wrlist[1].sendFlags = 0;
    qpHandle->rdmaOps = rdmaHandle.rdmaOps;
    ret = RaHdcSendWrlist(qpHandle, wrlist, opRspList, wrlistNum);
    EXPECT_INT_EQ(0, ret);

    struct SendWrlistDataExt dataExt[wrlistNum.sendNum];
    dataExt[0].memList = mem;
    dataExt[0].dstAddr = 0x111;
    dataExt[0].op = 0;
    dataExt[0].sendFlags = 0;
    dataExt[1].memList = mem;
    dataExt[1].dstAddr = 0x111;
    dataExt[1].op = 0;
    dataExt[1].sendFlags = 0;
    ret = RaHdcSendWrlistExt(qpHandle, dataExt, opRspList, wrlistNum);
    EXPECT_INT_EQ(0, ret);

    mocker((stub_fn_t)RaHdcLitePostSend, 10, -12);
    ret = RaHdcSendWrlist(qpHandle, wrlist, opRspList, wrlistNum);
    EXPECT_INT_EQ(ret, -12);

    mocker((stub_fn_t)RaHdcLitePostSend, 10, -12);
    ret = RaHdcSendWrlistExt(qpHandle, dataExt, opRspList, wrlistNum);
    EXPECT_INT_EQ(ret, -12);

    RaHdcLitePeriodPollCqe(&rdmaHandle);

    qpHandle->qpMode = RA_RS_OP_QP_MODE;
    qpHandle->sqDepth = 256;
    qpHandle->sendWrNum = 256;
    qpHandle->pollCqeNum = 0;
    ret = RaHdcSendWrV2(qpHandle, &wrV2, &rsp);
    EXPECT_INT_EQ(ret, -12);
    ret = RaHdcSendWrlistExt(qpHandle, dataExt, opRspList, wrlistNum);
    EXPECT_INT_EQ(ret, -12);
    qpHandle->pollCqeNum = 2;
    ret = RaHdcSendWrV2(qpHandle, &wrV2, &rsp);
    EXPECT_INT_EQ(ret, -12);
    ret = RaHdcSendWrlistExt(qpHandle, dataExt, opRspList, wrlistNum);
    EXPECT_INT_EQ(ret, -12);
    qpHandle->sendWrNum = 0;
    qpHandle->pollCqeNum = -256;
    ret = RaHdcSendWrV2(qpHandle, &wrV2, &rsp);
    EXPECT_INT_EQ(ret, -12);
    ret = RaHdcSendWrlistExt(qpHandle, dataExt, opRspList, wrlistNum);
    EXPECT_INT_EQ(ret, -12);

    ret = RaHdcQpDestroy(qpHandle);
    EXPECT_INT_EQ(ret, 0);

    RaHdcRdevDeinit(&rdmaHandle, NOTIFY);

    TcHdcTestEnvDeinit();

    free(addr);
    mocker_clean();
}

void TcHdcRecvWrlist()
{
    mocker_clean();
    void* addr = NULL;
    int size = 0;
    int ret = 0;
    struct RecvWrlistData revWr = {0};
    unsigned int recvNum = 1;
    unsigned int revCompleteNum = 0;
    TcHdcTestEnvInit();
    struct rdev rdevInfo = {0};
    unsigned int rdevIndex = 0;
    struct RaRdmaHandle rdmaHandle = {0};
    struct RaQpHandle* qpHandle = NULL;
    struct RaQpHandle qpHandleTmp = {0};

    revWr.wrId = 100;
    revWr.memList.lkey = 0xff;
    revWr.memList.addr = (uint64_t)(uintptr_t)addr;
    revWr.memList.len = size;

    qpHandleTmp.qpMode = 0;
    ret = RaHdcRecvWrlist(&qpHandleTmp, &revWr, recvNum, &revCompleteNum);
    EXPECT_INT_NE(ret, 0);

    mocker_invoke((stub_fn_t)RaHdcGetOpcodeLiteSupport, (stub_fn_t)StubRaHdcGetOpcodeLiteSupport, 100);
    mocker_invoke((stub_fn_t)RaHdcProcessMsg, (stub_fn_t)StubRaHdcProcessRdevInit, 100);
    ret = RaHdcRdevInit(&rdmaHandle, NOTIFY, rdevInfo, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);
    ret = RaHdcQpCreate(&rdmaHandle, 0, 2, (void**)&qpHandle);
    EXPECT_INT_EQ(ret, 0);
    ASSERT_ADDR_NE(qpHandle, NULL);
    qpHandle->supportLite = 1;

    ret = RaHdcRecvWrlist(qpHandle, &revWr, recvNum, &revCompleteNum);
    EXPECT_INT_EQ(ret, 0);

    ret = RaHdcQpDestroy(qpHandle);
    EXPECT_INT_EQ(ret, 0);
    RaHdcRdevDeinit(&rdmaHandle, NOTIFY);
    TcHdcTestEnvDeinit();
    mocker_clean();
}

void TcHdcPollCq()
{
    mocker_clean();
    int ret = 0;
    unsigned int numEntries = 1;
    struct rdma_lite_wc_v2 liteWc = {0};
    TcHdcTestEnvInit();
    struct rdev rdevInfo = {0};
    unsigned int rdevIndex = 0;
    struct RaRdmaHandle rdmaHandle = {0};
    struct RaQpHandle* qpHandle = NULL;
    struct RaQpHandle qpHandleTmp = {0};

    qpHandleTmp.qpMode = 0;
    ret = RaHdcPollCq(&qpHandleTmp, true, numEntries, &liteWc);
    EXPECT_INT_NE(ret, 0);

    mocker_invoke((stub_fn_t)RaHdcGetOpcodeLiteSupport, (stub_fn_t)StubRaHdcGetOpcodeLiteSupport, 100);
    mocker_invoke((stub_fn_t)RaHdcProcessMsg, (stub_fn_t)StubRaHdcProcessRdevInit, 100);
    rdmaHandle.disabledLiteThread = true;
    ret = RaHdcRdevInit(&rdmaHandle, NOTIFY, rdevInfo, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);
    ret = RaHdcQpCreate(&rdmaHandle, 0, 2, (void**)&qpHandle);
    EXPECT_INT_EQ(ret, 0);
    ASSERT_ADDR_NE(qpHandle, NULL);
    qpHandle->supportLite = 1;
    qpHandle->recvWrNum = 1;

    ret = RaHdcPollCq(qpHandle, false, numEntries, &liteWc);
    EXPECT_INT_EQ(ret, 0);

    ret = RaHdcQpDestroy(qpHandle);
    EXPECT_INT_EQ(ret, 0);
    RaHdcRdevDeinit(&rdmaHandle, NOTIFY);
    TcHdcTestEnvDeinit();
    mocker_clean();
}

void TcHdcGetLiteSupport()
{
    int support;

    gInterfaceVersion = 0;
    mocker_invoke(RaHdcGetInterfaceVersion, StubRaGetInterfaceVersion, 100);
    RaHdcGetOpcodeLiteSupport(0, 0x3, &support);
    EXPECT_INT_EQ(support, 0);
    mocker_clean();

    gInterfaceVersion = 2;
    mocker_invoke(RaHdcGetInterfaceVersion, StubRaGetInterfaceVersion, 100);
    RaHdcGetOpcodeLiteSupport(0, 0x3, &support);
    EXPECT_INT_EQ(support, 1);
    mocker_clean();

    gInterfaceVersion = 2;
    mocker_invoke(RaHdcGetInterfaceVersion, StubRaGetInterfaceVersion, 100);
    RaHdcGetOpcodeLiteSupport(0, 0x2, &support);
    EXPECT_INT_EQ(support, 2);
    mocker_clean();

    gInterfaceVersion = 1;
    mocker_invoke(RaHdcGetInterfaceVersion, StubRaGetInterfaceVersion, 100);
    RaHdcGetOpcodeLiteSupport(0, 0x2, &support);
    EXPECT_INT_EQ(support, 0);
    mocker_clean();
}

void TcRaRdevGetSupportLite()
{
    struct RaRdmaHandle rdmaHandle = {0};
    int supportLite = 1;
    int ret;

    ret = RaRdevGetSupportLite(NULL, 0);
    EXPECT_INT_NE(ret, 0);

    ret = RaRdevGetSupportLite(&rdmaHandle, &supportLite);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(supportLite, rdmaHandle.supportLite);
}

void TcRaRdevGetHandle()
{
    void* rdmaHandle = NULL;
    int ret;

    ret = RaRdevGetHandle(1024, 0);
    EXPECT_INT_EQ(ret, -EINVAL);

    ret = RaRdevGetHandle(0, 0);
    EXPECT_INT_EQ(ret, -EINVAL);

    ret = RaRdevGetHandle(0, &rdmaHandle);
    EXPECT_INT_EQ(ret, -ENODEV);
}

void TcRaIsFirstOrLastUsed()
{
    s32 ret = 0;

    ret = RaIsFirstUsed(-1);
    EXPECT_INT_EQ(ret, -EINVAL);

    ret = RaIsFirstUsed(0);
    EXPECT_INT_EQ(ret, 1);

    ret = RaIsFirstUsed(0);
    EXPECT_INT_EQ(ret, 0);

    ret = RaIsFirstUsed(0);
    EXPECT_INT_EQ(ret, 0);

    ret = RaIsLastUsed(-1);
    EXPECT_INT_EQ(ret, -EINVAL);

    ret = RaIsLastUsed(0);
    EXPECT_INT_EQ(ret, 0);

    ret = RaIsLastUsed(0);
    EXPECT_INT_EQ(ret, 0);

    ret = RaIsLastUsed(0);
    EXPECT_INT_EQ(ret, 1);

    ret = RaIsLastUsed(0);
    EXPECT_INT_EQ(ret, -EINVAL);

    ret = RaIsLastUsed(128);
    EXPECT_INT_EQ(ret, -EINVAL);

    ret = RaIsFirstUsed(128);
    EXPECT_INT_EQ(ret, -EINVAL);

    ret = RaIsFirstUsed(15);
    EXPECT_INT_EQ(ret, 1);

    ret = RaIsLastUsed(15);
    EXPECT_INT_EQ(ret, 1);
}

void TcRaHdcLiteCtxInit()
{
    struct RaRdmaHandle rdmaHandle = {0};
    unsigned int phyId = 0;
    unsigned int rdevIndex = 0;
    struct rdma_lite_context rdmaLiteContext = {0};
    int ret = 0;

    rdmaHandle.supportLite = 2 * 1024 * 1024;
    mocker_clean();
    mocker(RaHdcLiteMutexDeinit, 10, 0);
    mocker(RaRdmaLiteFreeCtx, 10, 0);
    mocker(DlHalSensorNodeUnregister, 10, 0);
    mocker(DlDrvDeviceGetIndexByPhyId, 10, 0);
    mocker(RaSensorNodeRegister, 10, 0);
    mocker(RaHdcProcessMsg, 10, 0);
    mocker(RaRdmaLiteAllocCtx, 1, NULL);
    ret = RaHdcLiteCtxInit(&rdmaHandle, phyId, rdevIndex);
    EXPECT_INT_EQ(ret, -14);

    mocker_clean();
    mocker(RaHdcLiteMutexDeinit, 10, 0);
    mocker(RaRdmaLiteFreeCtx, 10, 0);
    mocker(DlHalSensorNodeUnregister, 10, 0);
    mocker(DlDrvDeviceGetIndexByPhyId, 10, 0);
    mocker(RaSensorNodeRegister, 10, 0);
    mocker(RaHdcProcessMsg, 10, 0);
    mocker(RaRdmaLiteAllocCtx, 10, &rdmaLiteContext);
    mocker(RaHdcLiteMutexInit, 10, 0);
    rdmaHandle.disabledLiteThread = true;
    ret = RaHdcLiteCtxInit(&rdmaHandle, phyId, rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    rdmaHandle.disabledLiteThread = false;
    mocker(pthread_create, 10, -1);
    ret = RaHdcLiteCtxInit(&rdmaHandle, phyId, rdevIndex);
    EXPECT_INT_EQ(ret, -258);

    mocker_clean();
    mocker(RaHdcLiteMutexDeinit, 10, 0);
    mocker(RaRdmaLiteFreeCtx, 10, 0);
    mocker(DlHalSensorNodeUnregister, 10, 0);
    mocker(DlDrvDeviceGetIndexByPhyId, 10, 0);
    mocker(RaSensorNodeRegister, 10, 0);
    mocker(RaHdcProcessMsg, 10, 0);
    mocker(RaRdmaLiteAllocCtx, 10, &rdmaLiteContext);
    mocker(RaHdcLiteMutexInit, 10, 0);
    mocker(pthread_create, 10, 0);
    ret = RaHdcLiteCtxInit(&rdmaHandle, phyId, rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    mocker_clean();
    mocker(pthread_mutex_init, 10, 0);
    ret = RaHdcLiteMutexInit(&rdmaHandle, phyId);
    EXPECT_INT_EQ(ret, 0);

    mocker_clean();
    mocker(pthread_mutex_init, 1, -1);
    ret = RaHdcLiteMutexInit(&rdmaHandle, phyId);
    EXPECT_INT_EQ(ret, -258);
}

struct rdma_lite_cq* stub_RaRdmaLiteCreateCq(struct rdma_lite_context* liteCtx, struct rdma_lite_cq_attr* liteCqAttr)
{
    static int cnt = 0;
    static struct rdma_lite_cq liteCq = {0};

    cnt++;
    if (cnt == 1) {
        return NULL;
    } else {
        return &liteCq;
    }
}

void RcRaHdcLiteQpCreate()
{
    struct RaQpHandle qpHdc = {0};
    struct rdma_lite_qp_cap cap = {0};
    struct rdma_lite_cq liteCq = {0};
    struct rdma_lite_qp liteQp = {0};
    struct RaRdmaHandle rdmaHandle = {0};
    struct rdma_lite_cq_attr liteSendCqAttr = {0};
    struct rdma_lite_cq_attr liteRecvCqAttr = {0};
    struct rdma_lite_qp_attr liteQpAttr = {0};
    struct rdma_lite_wc liteWc = {0};
    unsigned int apiVersion = 0;
    int ret = 0;

    qpHdc.list.next = &(qpHdc.list);
    qpHdc.list.prev = &(qpHdc.list);
    rdmaHandle.qpList.next = &(rdmaHandle.qpList);
    rdmaHandle.qpList.prev = &(rdmaHandle.qpList);
    qpHdc.supportLite = 1;
    qpHdc.qpMode = 2;
    rdmaHandle.supportLite = 1;
    cap.max_inline_data = QP_DEFAULT_MAX_CAP_INLINE_DATA;
    cap.max_send_sge = QP_DEFAULT_MIN_CAP_SEND_SGE;
    cap.max_recv_sge = 1;
    cap.max_send_wr = RA_QP_TX_DEPTH;
    cap.max_recv_wr = RA_QP_TX_DEPTH;

    mocker_clean();
    mocker(RaHdcLiteGetCqQpAttr, 10, 0);
    mocker(RaHdcLiteInitMemPool, 10, 0);
    mocker(RaRdmaLiteDestroyCq, 10, 0);
    mocker(RaHdcLiteDeinitMemPool, 10, 0);
    mocker(RaRdmaLiteCreateCq, 1, NULL);
    ret = RaHdcLiteQpCreate(&rdmaHandle, &qpHdc, &cap);
    EXPECT_INT_EQ(ret, -14);

    mocker_clean();
    mocker(RaHdcLiteGetCqQpAttr, 10, 0);
    mocker(RaHdcLiteInitMemPool, 10, 0);
    mocker(RaRdmaLiteDestroyCq, 10, 0);
    mocker(RaHdcLiteDeinitMemPool, 10, 0);
    mocker_invoke(RaRdmaLiteCreateCq, stub_RaRdmaLiteCreateCq, 2);
    ret = RaHdcLiteQpCreate(&rdmaHandle, &qpHdc, &cap);
    EXPECT_INT_EQ(ret, -14);

    mocker_clean();
    mocker(RaHdcLiteGetCqQpAttr, 10, 0);
    mocker(RaHdcLiteInitMemPool, 10, 0);
    mocker(RaRdmaLiteDestroyCq, 10, 0);
    mocker(RaHdcLiteDeinitMemPool, 10, 0);
    mocker(RaRdmaLiteCreateCq, 2, &liteCq);
    mocker(RaRdmaLiteCreateQp, 1, NULL);
    ret = RaHdcLiteQpCreate(&rdmaHandle, &qpHdc, &cap);
    EXPECT_INT_EQ(ret, -14);

    mocker_clean();
    mocker(RaHdcLiteGetCqQpAttr, 10, 0);
    mocker(RaHdcLiteInitMemPool, 10, 0);
    mocker(RaRdmaLiteDestroyQp, 10, 0);
    mocker(RaRdmaLiteDestroyCq, 10, 0);
    mocker(RaHdcLiteDeinitMemPool, 10, 0);
    mocker(RaRdmaLiteCreateCq, 10, &liteCq);
    mocker(RaRdmaLiteCreateQp, 10, &liteQp);
    mocker(pthread_mutex_init, 10, 0);
    mocker(pthread_mutex_lock, 10, 0);
    mocker(pthread_mutex_unlock, 10, 0);
    mocker(calloc, 10, NULL);
    ret = RaHdcLiteQpCreate(&rdmaHandle, &qpHdc, &cap);
    EXPECT_INT_EQ(ret, -12);

    mocker_clean();
    mocker(RaHdcLiteGetCqQpAttr, 10, 0);
    mocker(RaHdcLiteInitMemPool, 10, 0);
    mocker(RaRdmaLiteDestroyCq, 10, 0);
    mocker(RaHdcLiteDeinitMemPool, 10, 0);
    mocker(RaRdmaLiteCreateCq, 10, &liteCq);
    mocker(RaRdmaLiteCreateQp, 10, &liteQp);
    mocker(pthread_mutex_init, 10, 0);
    mocker(pthread_mutex_lock, 10, 0);
    mocker(pthread_mutex_unlock, 10, 0);
    mocker(calloc, 10, &liteWc);
    ret = RaHdcLiteQpCreate(&rdmaHandle, &qpHdc, &cap);
    EXPECT_INT_EQ(ret, 0);

    mocker_clean();
    mocker(RaHdcProcessMsg, 10, 0);
    mocker(RaHdcLiteQpAttrInit, 10, 0);
    mocker(memcpy_s, 10, 0);
    ret = RaHdcLiteGetCqQpAttr(&qpHdc, &liteSendCqAttr, &liteRecvCqAttr, &liteQpAttr);
    EXPECT_INT_EQ(ret, 0);

    mocker_clean();
    mocker(RaHdcProcessMsg, 10, 0);
    mocker(RaRdmaLiteInitMemPool, 10, 0);
    ret = RaHdcLiteInitMemPool(&rdmaHandle, &qpHdc, &liteSendCqAttr, &liteRecvCqAttr, &liteQpAttr);
    EXPECT_INT_EQ(ret, 0);

    mocker_clean();
    mocker(RaRdmaLiteDeinitMemPool, 10, 0);
    RaHdcLiteDeinitMemPool(&rdmaHandle, &qpHdc);

    mocker_clean();
    ret = RaRdmaLiteGetApiVersion();
    EXPECT_INT_EQ(ret, 0);
}

void TcRaHdcTlvRequest()
{
    struct RaTlvHandle tlvHandleTmp = {0};
    struct TlvMsg sendMsg = {0};
    struct TlvMsg recvMsg = {0};
    unsigned int moduleType;
    int ret = 0;

    moduleType = TLV_MODULE_TYPE_CCU;
    sendMsg.length = 0;
    sendMsg.type = 0;
    mocker(RaHdcProcessMsg, 100, 0);
    gInterfaceVersion = 1;
    mocker_invoke(RaHdcGetInterfaceVersion, StubRaGetInterfaceVersion, 10);
    ret = RaHdcTlvRequest(&tlvHandleTmp, moduleType, &sendMsg, &recvMsg);
    EXPECT_INT_EQ(ret, 0);

    tlvHandleTmp.initInfo.phyId = 0;
    moduleType = TLV_MODULE_TYPE_NSLB;
    sendMsg.length = TC_TLV_HDC_MSG_SIZE;
    sendMsg.type = 0;
    sendMsg.data = (char*)malloc(TC_TLV_HDC_MSG_SIZE);
    int i = 0;
    for (i = 0; i < TC_TLV_HDC_MSG_SIZE; i++) {
        sendMsg.data[i] = (char)(i % 256);
    }

    mocker(RaHdcProcessMsg, 100, 0);
    gInterfaceVersion = 1;
    mocker_invoke(RaHdcGetInterfaceVersion, StubRaGetInterfaceVersion, 10);
    ret = RaHdcTlvRequest(&tlvHandleTmp, moduleType, &sendMsg, &recvMsg);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    free(sendMsg.data);
    sendMsg.data = NULL;
}

void TcRaHdcQpCreateWithAttrs()
{
    struct RaRdmaHandle rdmaHandle = {0};
    struct QpExtAttrs extAttrs = {0};
    struct AiQpInfo info = {0};
    void* qpHandle = NULL;
    int ret = 0;

    mocker(memcpy_s, 1, -1);
    ret = RaHdcQpCreateWithAttrs(&rdmaHandle, &extAttrs, &qpHandle);
    EXPECT_INT_EQ(ret, -256);
    mocker_clean();

    mocker(memcpy_s, 1, -1);
    ret = RaHdcAiQpCreate(&rdmaHandle, &extAttrs, &info, &qpHandle);
    EXPECT_INT_EQ(ret, -256);
    mocker_clean();

    mocker(memcpy_s, 1, -1);
    ret = RaHdcAiQpCreateWithAttrs(&rdmaHandle, &extAttrs, &info, &qpHandle);
    EXPECT_INT_EQ(ret, -256);
    mocker_clean();
}
