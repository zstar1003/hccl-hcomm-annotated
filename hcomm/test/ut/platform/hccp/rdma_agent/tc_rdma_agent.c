/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define _GNU_SOURCE
#include <sched.h>
#include <sys/mman.h>
#include "ra.h"
#include "ra_rs_err.h"
#include "ra_client_host.h"
#include "hccp.h"
#include "ut_dispatch.h"
#include "stdlib.h"
#include "securec.h"
#include <pthread.h>
#include "dlfcn.h"
#include "rs.h"
#include "dl.h"
#include "ra_hdc.h"
#include "ra_hdc_lite.h"
#include "ra_hdc_rdma_notify.h"
#include "ra_hdc_rdma.h"
#include "ra_async.h"
#include "ra_hdc_socket.h"
#include "dl_hal_function.h"
#include "ra_peer.h"
#include "ra_adp.h"
#include "ascend_hal.h"
#include <errno.h>
#include "ra_comm.h"
#include "ra_hdc_async.h"

extern int HdcSendRecvPkt(void* session, void* pSendRcvBuf, unsigned int inBufLen, unsigned int outDataLen);

extern int
RaPeerRdevInit(struct RaRdmaHandle* rdmaHandle, unsigned int notifyType, struct rdev rdevInfo, unsigned int* rdevIndex);
extern int RsRdevInit(struct rdev rdevInfo, unsigned int notifyType, unsigned int* rdevIndex);
extern int RaPeerGetServerDevid(int logicDevid, int* serverDevid);
extern int RsRdevDeinit(unsigned int devId, unsigned int notifyType, unsigned int rdevIndex);
extern int RaPeerSocketWhiteListAdd(struct rdev rdevInfo, struct SocketWlistInfoT whiteList[], unsigned int num);
extern int RsSocketWhiteListAdd(struct rdev rdevInfo, struct SocketWlistInfoT whiteList[], unsigned int num);
extern int RsSocketWhiteListDel(struct rdev rdevInfo, struct SocketWlistInfoT whiteList[], unsigned int num);
extern int RaGetSocketConnectInfo(
    const struct SocketConnectInfoT conn[], unsigned int num, struct SocketConnectInfo rsConn[], unsigned int rsNum);
extern int RaGetSocketListenResult(
    const struct SocketListenInfo rsConn[], unsigned int rsNum, struct SocketListenInfoT conn[], unsigned int num);
extern int RsSocketListenStart(struct SocketListenInfo conn[], uint32_t num);
extern int
RaPeerSetRsConnParam(struct SocketInfoT conn[], unsigned int num, struct SocketFdData rsConn[], unsigned int rsNum);
extern int RaInetPton(int family, union HccpIpAddr ip, char netAddr[], unsigned int len);
extern int RaHdcRdevDeinit(struct RaRdmaHandle* rdmaHandle, unsigned int notifyType);
extern int
RaHdcRdevInit(struct RaRdmaHandle* rdmaHandle, unsigned int notifyType, struct rdev rdevInfo, unsigned int* rdevIndex);
extern int RaHdcInitApart(int devId, unsigned int* phyId);
extern int MsgHeadCheck(struct MsgHead* sendRcvHead, unsigned int opcode, int rsRet, unsigned int msgDataLen);
extern int RaRdevInitCheckIp(int mode, struct rdev rdevInfo, char localIp[]);
extern int RaHdcGetLiteSupport(struct RaRdmaHandle* rdmaHandle, unsigned int phyId);
extern int RaHdcNotifyBaseAddrInit(unsigned int notifyType, unsigned int phyId, unsigned long long** notifyVa);
extern int RaHdcAsyncSessionClose(unsigned int phyId);
extern void RaHwAsyncHdcClientDeinit(unsigned int phyId);
extern void RaHdcAsyncMutexDeinit(unsigned int phyId);
extern int RaRdevGetHandle(unsigned int phyId, void** rdmaHandle);
extern int RaSaveSnapshot(struct RaInfo* info, enum SaveSnapshotAction action);
extern int RaRestoreSnapshot(struct RaInfo* info);
extern int RaHdcAsyncSessionConnect(struct RaInitConfig* cfg);
extern int RaHdcInitSession(int peerNode, int peerDevid, unsigned int phyId, int hdcType, HDC_SESSION* session);

extern void* mmap(void* start, size_t length, int prot, int flags, int fd, off_t offsize);
extern int munmap(void* start, size_t length);
extern int open(const char* pathname, int flags);
extern int ioctl(int fd, unsigned long cmd, struct HostRoceNotifyInfo* info);
extern hdcError_t
DlHalHdcRecv(HDC_SESSION session, struct drvHdcMsg* pMsg, int bufLen, UINT64 flag, int* recvBufCount, UINT32 timeout);
extern hdcError_t DlDrvHdcAllocMsg(HDC_SESSION session, struct drvHdcMsg** ppMsg, int count);
extern hdcError_t DlDrvHdcFreeMsg(struct drvHdcMsg* msg);
extern hdcError_t DlDrvHdcSessionClose(HDC_SESSION session);
extern int DlDrvDeviceGetIndexByPhyId(uint32_t phyId, uint32_t* devIndex);
extern int dlHalNotifyGetInfo(uint32_t devId, uint32_t tsId, uint32_t type, uint32_t* val);
extern int dlHalMemAlloc(void** pp, unsigned long long size, unsigned long long flag);
extern int gNotifyFd;

extern int RaRsGetIfnum(char* inBuf, char* outBuf, int* outLen, int* opResult, int rcvBufLen);
extern int RaRsSocketBatchConnect(char* inBuf, char* outBuf, int* outLen, int* opResult, unsigned int size);
extern int RaRsSocketListenStart(char* inBuf, char* outBuf, int* outLen, int* opResult, unsigned int size);
extern int RaRsSocketListenStop(char* inBuf, char* outBuf, int* outLen, int* opResult, unsigned int size);
extern int RaRsGetVnicIpInfosV1(char* inBuf, char* outBuf, int* outLen, int* opResult, unsigned int size);
extern int RaRsGetVnicIpInfos(char* inBuf, char* outBuf, int* outLen, int* opResult, unsigned int size);
extern int RaRsTypicalMrRegV1(char* inBuf, char* outBuf, int* outLen, int* opResult, int rcvBufLen);
extern int RaRsTypicalMrDereg(char* inBuf, char* outBuf, int* outLen, int* opResult, int rcvBufLen);
extern int RaRsTypicalMrReg(char* inBuf, char* outBuf, int* outLen, int* opResult, int rcvBufLen);
extern int RaRsTypicalQpCreate(char* inBuf, char* outBuf, int* outLen, int* opResult, int rcvBufLen);
extern int RaRsTypicalQpModify(char* inBuf, char* outBuf, int* outLen, int* opResult, int rcvBufLen);
extern int RaHdcRecvHandleSendPkt(unsigned int phyId);
extern int RaRsGetSecRandom(char* inBuf, char* outBuf, int* outLen, int* opResult, unsigned int size);
extern int RaRsGetTlsEnable(char* inBuf, char* outBuf, int* outLen, int* opResult, unsigned int size);
extern int RaRsRdevGetPortStatus(char* inBuf, char* outBuf, int* outLen, int* opResult, int rcvBufLen);
extern int RaRsGetHccnCfg(char* inBuf, char* outBuf, int* outLen, int* opResult, unsigned int size);
extern void RaHwHdcInit(void* arg);
extern void RaHwAsyncDelList(struct RaListHead* list, pthread_mutex_t* mutex);
extern void RaHdcUninitAsync(void);

int secCpyRet = 0;

#define MAX_DEV_NUM 8

void* StubCalloc(size_t nmemb, size_t size)
{
    static int i = 0;
    void* p = NULL;
    if (i == 0) {
        i++;
        p = (void*)malloc(nmemb * size);
        return p;
    } else {
        return NULL;
    }
}

static unsigned int gInterfaceVersion;

static int RaGetInterfaceVersionStub(unsigned int phyId, unsigned int interfaceOpcode, unsigned int* interfaceVersion)
{
    *interfaceVersion = gInterfaceVersion;
    return 0;
}

DLLEXPORT drvError_t StubSessionConnectHdc(int peerNode, int peerDevid, HDC_CLIENT client, HDC_SESSION* session)
{
    static HDC_SESSION gHdcSession = (HDC_SESSION)1;
    *session = gHdcSession;
    return 0;
}

void TcHdcEnvInit()
{
    struct RaInitConfig offlineHdcConfig = {
        .phyId = 0,
        .nicPosition = NETWORK_OFFLINE,
        .hdcType = HDC_SERVICE_TYPE_RDMA,
        .enableHdcAsync = false,
    };
    struct ProcessRaSign pRaSign;
    pRaSign.tgid = 0;

    mocker_clean();
    mocker((stub_fn_t)drvHdcClientCreate, 1, 0);
    mocker_invoke((stub_fn_t)drvHdcSessionConnect, (stub_fn_t)StubSessionConnectHdc, 1);
    mocker((stub_fn_t)drvHdcSetSessionReference, 1, 0);
    mocker((stub_fn_t)halHdcRecv, 10, 0);
    int ret = RaHdcInit(&offlineHdcConfig, pRaSign);
    EXPECT_INT_EQ(ret, 0);
}

void TcHdcEnvDeinit()
{
    struct RaInitConfig offlineHdcConfig = {
        .phyId = 0,
        .nicPosition = NETWORK_OFFLINE,
        .hdcType = HDC_SERVICE_TYPE_RDMA,
        .enableHdcAsync = false,
    };

    mocker((stub_fn_t)halHdcRecv, 10, 0);
    mocker((stub_fn_t)drvHdcSessionClose, 1, 0);
    mocker((stub_fn_t)drvHdcClientDestroy, 1, 0);
    int ret = RaHdcDeinit(&offlineHdcConfig);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

int RaHdcGetLiteSupportStub(struct RaRdmaHandle* rdmaHandle, unsigned int phyId)
{
    rdmaHandle->supportLite = 1;
    return 0;
}

void TcHostAbnormalQpModeTest()
{
    int ret;
    struct rdev rdevInfo = {0};
    rdevInfo.family = AF_INET;
    struct RaRdmaHandle* rdmaHandle = NULL;
    void* qpHandle = NULL;
    RaRdevInit(NETWORK_OFFLINE, NOTIFY, rdevInfo, (void**)&rdmaHandle);

    ret = RaQpCreate(rdmaHandle, 0, 3, &qpHandle);
    EXPECT_INT_NE(0, ret);
}

extern void RaHwInit(void* arg);

extern int
HdcSendRecvPktRecvCheck(int rcvBufLen, unsigned int outDataLen, struct MsgHead* recvMsgHead, struct drvHdcMsg* pMsgRcv);
void TcHdcSendRecvPktRecvCheck() {}

void TcRaPeerSocketWhiteListAdd01()
{
    struct rdev rdevInfo = {0};
    struct SocketWlistInfoT whiteList[4] = {0};
    RaPeerSocketWhiteListAdd(rdevInfo, whiteList, 1);
}

void TcRaPeerSocketWhiteListAdd02()
{
    struct rdev rdevInfo = {0};
    mocker(RsSocketWhiteListAdd, 20, 1);
    struct SocketWlistInfoT whiteList[4] = {0};
    RaPeerSocketWhiteListAdd(rdevInfo, whiteList, 1);
    mocker_clean();
}

void TcRaPeerSocketWhiteListDel()
{
    struct rdev rdevInfo = {0};
    struct SocketWlistInfoT whiteList[5] = {0};
    mocker(RsSocketWhiteListDel, 20, 1);
    RaPeerSocketWhiteListDel(rdevInfo, whiteList, 5);
    mocker_clean();
}

void TcRaPeerRdevInit01()
{
    int ret;
    struct rdev rdevInfo = {0};
    struct RaRdmaHandle rdmaHandle = {0};
    unsigned int* rdevIndex = (unsigned int*)malloc(sizeof(unsigned int));
    mocker(HostNotifyBaseAddrInit, 1, 0);
    mocker(RsRdevInit, 1, 0);
    ret = RaPeerRdevInit(&rdmaHandle, NOTIFY, rdevInfo, rdevIndex);
    mocker_clean();
    free(rdevIndex);
    EXPECT_INT_EQ(0, ret);
}

void TcRaPeerRdevInit02()
{
    int ret;
    struct rdev rdevInfo = {0};
    struct RaRdmaHandle rdmaHandle = {0};
    unsigned int* rdevIndex = (unsigned int*)malloc(sizeof(unsigned int));
    mocker(HostNotifyBaseAddrInit, 1, 1);
    ret = RaPeerRdevInit(&rdmaHandle, NOTIFY, rdevInfo, rdevIndex);
    mocker_clean();
    free(rdevIndex);
    EXPECT_INT_EQ(1, ret);
}

void TcRaPeerRdevInit03()
{
    int ret;
    struct rdev rdevInfo = {0};
    struct RaRdmaHandle rdmaHandle = {0};
    unsigned int* rdevIndex = (unsigned int*)malloc(sizeof(unsigned int));
    mocker(HostNotifyBaseAddrInit, 1, 0);
    mocker(RsRdevInit, 1, 1);
    mocker(HostNotifyBaseAddrUninit, 1, 0);
    ret = RaPeerRdevInit(&rdmaHandle, NOTIFY, rdevInfo, rdevIndex);
    mocker_clean();
    free(rdevIndex);
    EXPECT_INT_EQ(1, ret);
}

void TcRaPeerRdevInit04()
{
    int ret;
    struct rdev rdevInfo = {0};
    struct RaRdmaHandle rdmaHandle = {0};
    unsigned int* rdevIndex = (unsigned int*)malloc(sizeof(unsigned int));
    mocker(HostNotifyBaseAddrInit, 1, 0);
    mocker(RsRdevInit, 1, 1);
    mocker(HostNotifyBaseAddrUninit, 1, 2);
    ret = RaPeerRdevInit(&rdmaHandle, NOTIFY, rdevInfo, rdevIndex);
    mocker_clean();
    free(rdevIndex);
    EXPECT_INT_EQ(2, ret);
}

void TcRaPeerRdevDeinit01()
{
    int ret;
    struct RaRdmaHandle* rdmaHandle = (struct RaRdmaHandle*)malloc(sizeof(struct RaRdmaHandle));
    rdmaHandle->rdevInfo.phyId = 0;
    mocker(RsRdevDeinit, 1, 0);
    mocker(HostNotifyBaseAddrUninit, 1, 0);
    ret = RaPeerRdevDeinit(rdmaHandle, NOTIFY);
    mocker_clean();
    free(rdmaHandle);
    rdmaHandle = NULL;
    EXPECT_INT_EQ(0, ret);
}

void TcRaPeerRdevDeinit02()
{
    int ret;
    struct RaRdmaHandle* rdmaHandle = (struct RaRdmaHandle*)malloc(sizeof(struct RaRdmaHandle));
    rdmaHandle->rdevInfo.phyId = 0;
    mocker(RsRdevDeinit, 1, 1);
    ret = RaPeerRdevDeinit(rdmaHandle, NOTIFY);
    mocker_clean();
    free(rdmaHandle);
    rdmaHandle = NULL;
    EXPECT_INT_EQ(1, ret);
}

void TcRaPeerRdevDeinit03()
{
    int ret;
    struct RaRdmaHandle* rdmaHandle = (struct RaRdmaHandle*)malloc(sizeof(struct RaRdmaHandle));
    rdmaHandle->rdevInfo.phyId = 0;
    mocker(RsRdevDeinit, 1, 0);
    mocker(HostNotifyBaseAddrUninit, 1, 2);
    ret = RaPeerRdevDeinit(rdmaHandle, NOTIFY);
    mocker_clean();
    free(rdmaHandle);
    rdmaHandle = NULL;
    EXPECT_INT_EQ(2, ret);
}

static int StubRsRdevGetPortStatusActive(unsigned int phyId, unsigned int rdevIndex, enum PortStatus* status)
{
    *status = PORT_STATUS_ACTIVE;
    return 0;
}

static int StubRsRdevGetPortStatusFail(unsigned int phyId, unsigned int rdevIndex, enum PortStatus* status)
{
    return -EINVAL;
}

void TcRaPeerRdevGetPortStatusSucc()
{
    int ret;
    enum PortStatus status = PORT_STATUS_DOWN;
    struct RaRdmaHandle* rdmaHandle = (struct RaRdmaHandle*)malloc(sizeof(struct RaRdmaHandle));
    rdmaHandle->rdevInfo.phyId = 0;

    mocker_invoke(RsRdevGetPortStatus, StubRsRdevGetPortStatusActive, 1);
    ret = RaPeerRdevGetPortStatus(rdmaHandle, &status);
    mocker_clean();
    free(rdmaHandle);
    rdmaHandle = NULL;
    EXPECT_INT_EQ(0, ret);
    EXPECT_INT_EQ(PORT_STATUS_ACTIVE, status);
}

void TcRaPeerRdevGetPortStatusFail()
{
    int ret;
    enum PortStatus status = PORT_STATUS_DOWN;
    struct RaRdmaHandle* rdmaHandle = (struct RaRdmaHandle*)malloc(sizeof(struct RaRdmaHandle));
    rdmaHandle->rdevInfo.phyId = 0;

    mocker_invoke(RsRdevGetPortStatus, StubRsRdevGetPortStatusFail, 1);
    ret = RaPeerRdevGetPortStatus(rdmaHandle, &status);
    mocker_clean();
    free(rdmaHandle);
    rdmaHandle = NULL;
    EXPECT_INT_EQ(-EINVAL, ret);
    EXPECT_INT_EQ(PORT_STATUS_DOWN, status);
}

void TcRaPeerSocketBatchConnect()
{
    unsigned int devId = 0;
    struct SocketConnectInfoT conn[4] = {0};
    mocker(RaGetSocketConnectInfo, 20, 1);
    RaPeerSocketBatchConnect(devId, conn, 5);
    mocker_clean();
}

void TcRaPeerSocketBatchAbort()
{
    unsigned int devId = 0;
    struct SocketConnectInfoT conn[4] = {0};
    int ret = 0;

    mocker(RaGetSocketConnectInfo, 20, 1);
    ret = RaPeerSocketBatchAbort(devId, conn, 5);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker(RaGetSocketConnectInfo, 20, 0);
    mocker(pthread_mutex_lock, 10, 0);
    mocker(pthread_mutex_unlock, 10, 0);
    mocker(RsSocketBatchAbort, 10, 1);
    ret = RaPeerSocketBatchAbort(devId, conn, 5);
    EXPECT_INT_EQ(ret, 1);
    mocker_clean();

    mocker(RaGetSocketConnectInfo, 20, 0);
    mocker(pthread_mutex_lock, 10, 0);
    mocker(pthread_mutex_unlock, 10, 0);
    mocker(RsSocketBatchAbort, 10, 0);
    ret = RaPeerSocketBatchAbort(devId, conn, 5);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRaPeerSocketListenStart01()
{
    unsigned int devId = 0;
    struct SocketListenInfoT conn[5] = {0};
    mocker(RaGetSocketListenInfo, 10, 1);
    RaPeerSocketListenStart(devId, conn, 5);
    mocker_clean();
}

void TcRaPeerSocketListenStart02()
{
    unsigned int devId = 0;
    struct SocketListenInfoT conn[5] = {0};
    mocker(RaGetSocketListenInfo, 10, 0);
    mocker(RsSocketListenStart, 10, 0);
    mocker(RaGetSocketListenResult, 10, 1);
    mocker_clean();
}

void TcRaPeerSocketListenStop()
{
    unsigned int devId = 0;
    struct SocketListenInfoT conn[5] = {0};
    mocker(RaGetSocketListenInfo, 10, 1);
    RaPeerSocketListenStop(devId, conn, 5);
    mocker_clean();
}

void TcRaPeerSetRsConnParam()
{
    struct SocketInfoT conn[6] = {0};
    struct SocketFdData rsConn[5] = {0};
    RaPeerSetRsConnParam(conn, 6, rsConn, 5);
}

void TcRaInetPton01()
{
    char netAddr[5] = {0};
    union HccpIpAddr ip = {0};
    RaInetPton(0, ip, netAddr, 32);
}

void TcRaInetPton02()
{
    char netAddr[5] = {0};
    union HccpIpAddr ip = {0};
    RaInetPton(2, ip, netAddr, 0);
}

void TcRaSocketInit()
{
    struct rdev rdevInfo = {0};
    void* socketHandle = NULL;

    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = 0;
    RaSocketInit(NETWORK_OFFLINE, rdevInfo, &socketHandle);
}

void TcRaSocketInitV1()
{
    struct SocketInitInfoT socketInit = {0};
    void* socketHandle = NULL;

    socketInit.scopeId = 0;
    socketInit.rdevInfo.phyId = 0;
    socketInit.rdevInfo.family = AF_INET;
    socketInit.rdevInfo.localIp.addr.s_addr = 0;
    RaSocketInitV1(NETWORK_OFFLINE, socketInit, &socketHandle);

    socketInit.scopeId = 0;
    socketInit.rdevInfo.phyId = 0;
    socketInit.rdevInfo.family = AF_INET6;
    socketInit.rdevInfo.localIp.addr.s_addr = 0;
    RaSocketInitV1(NETWORK_PEER_ONLINE, socketInit, &socketHandle);
    RaSocketDeinit(socketHandle);

    RaSocketInitV1(NETWORK_ONLINE, socketInit, &socketHandle);

    mocker(calloc, 1, NULL);
    RaSocketInitV1(NETWORK_PEER_ONLINE, socketInit, &socketHandle);
    mocker_clean();

    mocker(RaInetPton, 1, 99);
    RaSocketInitV1(NETWORK_PEER_ONLINE, socketInit, &socketHandle);
    mocker_clean();

    mocker(memcpy_s, 1, 1);
    RaSocketInitV1(NETWORK_PEER_ONLINE, socketInit, &socketHandle);
    mocker_clean();

    RaSocketInitV1(NETWORK_PEER_ONLINE, socketInit, NULL);
}

void TcRaSendWrlist()
{
    struct RaQpHandle qpHandle = {0};
    struct RaRdmaOps rdmaOps = {0};
    qpHandle.rdmaOps = &rdmaOps;
    unsigned int sendNum = 1;
    unsigned int completeNum = 0;
    struct SendWrlistData wrlist[1] = {{0}};
    struct SendWrRsp opRsp[1] = {{0}};
    RaSendWrlist(NULL, NULL, NULL, sendNum, &completeNum);
    qpHandle.rdmaOps = NULL;
    RaSendWrlist(&qpHandle, wrlist, opRsp, sendNum, &completeNum);
    wrlist[0].memList.len = 2147483649;
    RaSendWrlist(&qpHandle, wrlist, opRsp, sendNum, &completeNum);
}

void TcRaRdevInit()
{
    struct rdev rdevInfo = {0};
    void* rdmaHandle = NULL;
    rdevInfo.phyId = 0;
    RaRdevInit(2, NOTIFY, rdevInfo, &rdmaHandle);
}

void TcRaRdevGetPortStatus()
{
    enum PortStatus status = PORT_STATUS_DOWN;
    struct RaRdmaHandle rdmaHandle = {0};
    struct RaRdmaOps ops = {0};
    int ret;

    ret = RaRdevGetPortStatus(NULL, NULL);
    EXPECT_INT_NE(0, ret);

    rdmaHandle.rdevInfo.phyId = 100000;
    ret = RaRdevGetPortStatus(&rdmaHandle, &status);
    EXPECT_INT_NE(0, ret);

    rdmaHandle.rdevInfo.phyId = 0;
    ret = RaRdevGetPortStatus(&rdmaHandle, &status);
    EXPECT_INT_NE(0, ret);

    ops.raRdevGetPortStatus = RaHdcRdevGetPortStatus;
    rdmaHandle.rdmaOps = &ops;
    mocker(RaHdcProcessMsg, 5, -1);
    ret = RaRdevGetPortStatus(&rdmaHandle, &status);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    mocker(RaHdcProcessMsg, 5, 0);
    ret = RaRdevGetPortStatus(&rdmaHandle, &status);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    int outLen = 0;
    int opResult = 0;
    int rcvBufLen = 300;

    char inBuf[512];
    char outBuf[512];

    ret = RaRsRdevGetPortStatus(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(ret, 0);
}

void TcRaHdcRdevDeinit()
{
    struct RaRdmaHandle rdmaHandle = {0};
    mocker(calloc, 10, NULL);
    mocker(rdma_lite_free_context, 10, 0);
    int ret = RaHdcRdevDeinit(&rdmaHandle, NOTIFY);
    EXPECT_INT_EQ(-ENOMEM, ret);
    mocker_clean();

    mocker(HdcSendRecvPkt, 20, 0);
    mocker(MsgHeadCheck, 20, 1);
    mocker(rdma_lite_free_context, 10, 0);
    ret = RaHdcRdevDeinit(&rdmaHandle, NOTIFY);
    mocker_clean();
}

void TcRaHdcSocketWhiteListAdd()
{
    struct rdev rdevInfo = {0};
    struct SocketWlistInfoT whiteList[1] = {{0}};
    mocker(HdcSendRecvPkt, 20, 1);
    RaHdcSocketWhiteListAdd(rdevInfo, whiteList, 1);
    mocker_clean();

    mocker(HdcSendRecvPkt, 20, 0);
    mocker(MsgHeadCheck, 20, 1);
    RaHdcSocketWhiteListAdd(rdevInfo, whiteList, 1);
    mocker_clean();
}

void TcRaHdcSocketWhiteListDel()
{
    struct rdev rdevInfo = {0};
    struct SocketWlistInfoT whiteList[1] = {{0}};
    int ret;

    mocker(HdcSendRecvPkt, 20, 1);
    ret = RaHdcSocketWhiteListDel(rdevInfo, whiteList, 1);
    EXPECT_INT_EQ(1, ret);
    mocker_clean();

    mocker(HdcSendRecvPkt, 20, 0);
    mocker(MsgHeadCheck, 20, 1);
    ret = RaHdcSocketWhiteListDel(rdevInfo, whiteList, 1);
    EXPECT_INT_EQ(1, ret);
    mocker_clean();
}

void TcRaHdcSocketAcceptCreditAdd()
{
    struct SocketListenInfoT conn[1] = {{0}};
    struct RaSocketHandle socketHandle = {0};
    conn[0].socketHandle = &socketHandle;

    int ret;
    mocker(RaGetSocketListenInfo, 1, -1);
    ret = RaHdcSocketAcceptCreditAdd(1, conn, 1, 1);
    EXPECT_INT_EQ(1, 1);
    mocker_clean();

    mocker(RaHdcProcessMsg, 1, -1);
    ret = RaHdcSocketAcceptCreditAdd(1, conn, 1, 1);
    EXPECT_INT_EQ(1, 1);
    mocker_clean();
}

void TcRaHdcRdevInit()
{
    struct rdev rdevInfo = {0};
    unsigned int rdevIndex = 0;
    int ret;
    struct RaRdmaHandle rdmaHandle = {0};
    mocker(DlDrvDeviceGetIndexByPhyId, 20, 0);
    mocker(DlHalNotifyGetInfo, 20, 0);
    mocker(DlHalMemAlloc, 20, 0);
    mocker(calloc, 20, NULL);
    ret = RaHdcRdevInit(&rdmaHandle, NOTIFY, rdevInfo, &rdevIndex);
    EXPECT_INT_EQ(-ENOMEM, ret);
    mocker_clean();

    mocker(memcpy_s, 20, 1);
    mocker(DlDrvDeviceGetIndexByPhyId, 20, 0);
    mocker(DlHalNotifyGetInfo, 20, 0);
    mocker(DlHalMemAlloc, 20, 0);
    ret = RaHdcRdevInit(&rdmaHandle, NOTIFY, rdevInfo, &rdevIndex);
    EXPECT_INT_EQ(-ESAFEFUNC, ret);
    mocker_clean();

    mocker(HdcSendRecvPkt, 20, 0);
    mocker(MsgHeadCheck, 20, 1);
    mocker(DlDrvDeviceGetIndexByPhyId, 20, 0);
    mocker(DlHalNotifyGetInfo, 20, 0);
    mocker(DlHalMemAlloc, 20, 0);
    ret = RaHdcRdevInit(&rdmaHandle, NOTIFY, rdevInfo, &rdevIndex);
    EXPECT_INT_EQ(1, ret);
    mocker_clean();
}

void TcRaHdcInitApart()
{
    unsigned int phyId = 0;
    mocker(DlDrvDeviceGetIndexByPhyId, 20, 0);
    mocker(pthread_mutex_init, 20, 1);
    int ret = RaHdcInitApart(1, &phyId);
    EXPECT_INT_EQ(-ESYSFUNC, ret);
    mocker_clean();
}

void TcRaHdcQpDestroy()
{
    struct RaQpHandle* qpHdc = (struct RaQpHandle*)malloc(sizeof(struct RaQpHandle));
    *qpHdc = (struct RaQpHandle){0};
    mocker(HdcSendRecvPkt, 20, 1);
    mocker(rdma_lite_destroy_qp, 20, 0);
    mocker(rdma_lite_destroy_cq, 20, 0);
    int ret = RaHdcQpDestroy(qpHdc);
    EXPECT_INT_EQ(1, ret);
    mocker_clean();
}
void TcRaHdcQpDestroy01()
{
    struct RaQpHandle* qpHdc = (struct RaQpHandle*)malloc(sizeof(struct RaQpHandle));
    *qpHdc = (struct RaQpHandle){0};
    mocker(HdcSendRecvPkt, 20, 0);
    mocker(MsgHeadCheck, 20, 1);
    mocker(rdma_lite_destroy_qp, 20, 0);
    mocker(rdma_lite_destroy_cq, 20, 0);
    int ret = RaHdcQpDestroy(qpHdc);
    EXPECT_INT_EQ(1, ret);
    mocker_clean();
}

void TcRaGetSocketConnectInfo()
{
    int ret = RaGetSocketConnectInfo(NULL, 1, NULL, 1);
    EXPECT_INT_EQ(-EINVAL, ret);
}

void TcRaGetSocketListenInfo()
{
    int ret = RaGetSocketListenInfo(NULL, 1, NULL, 1);
    EXPECT_INT_EQ(-EINVAL, ret);
}

void TcRaGetSocketListenResult()
{
    int ret = RaGetSocketListenResult(NULL, 1, NULL, 1);
    EXPECT_INT_EQ(-EINVAL, ret);
}

void TcRaHwHdcInit()
{
    mocker((stub_fn_t)pthread_detach, 1, 0);
    mocker((stub_fn_t)pthread_create, 1, -1);
    RaHwHdcInit(NULL);
    mocker_clean();
}

void TcRaPeerInitFail001()
{
    struct RaInitConfig cfg = {0};
    unsigned int whiteListStatus = 0;

    mocker(pthread_mutex_init, 1, 1);
    int ret = RaPeerInit(&cfg, whiteListStatus);
    mocker_clean();
    EXPECT_INT_EQ(-ESYSFUNC, ret);
}

void TcRaPeerSocketDeinit001()
{
    struct rdev rdevInfo = {0};

    mocker(RsSocketDeinit, 1, 0);
    int ret = RaPeerSocketDeinit(rdevInfo);
    mocker_clean();
    EXPECT_INT_EQ(0, ret);
}

void TcHostNotifyBaseAddrInit()
{
    int ret;

    mocker(drvDeviceGetIndexByPhyId, 1, 0);
    mocker(halNotifyGetInfo, 1, 0);
    mocker(open, 1, 1);
    mocker(mmap, 1, 1);
    mocker(RsNotifyCfgSet, 1, 0);
    ret = HostNotifyBaseAddrInit(0);
    mocker_clean();
    EXPECT_INT_EQ(0, ret);
}

void TcHostNotifyBaseAddrInit001()
{
    int ret;

    mocker(drvDeviceGetIndexByPhyId, 1, 1);
    ret = HostNotifyBaseAddrInit(0);
    mocker_clean();
    EXPECT_INT_EQ(-1, ret);
}

void TcHostNotifyBaseAddrInit002()
{
    int ret;

    mocker(drvDeviceGetIndexByPhyId, 1, 0);
    mocker(halNotifyGetInfo, 1, 2);
    mocker(RsNotifyCfgSet, 1, 0);
    ret = HostNotifyBaseAddrInit(0);
    mocker_clean();
    EXPECT_INT_EQ(-2, ret);
}

void TcHostNotifyBaseAddrInit003()
{
    int ret;

    mocker(drvDeviceGetIndexByPhyId, 1, 0);
    mocker(halNotifyGetInfo, 1, 0);
    mocker(open, 1, -1);
    mocker(mmap, 1, MAP_FAILED);
    ret = HostNotifyBaseAddrInit(0);
    EXPECT_INT_EQ(-ENOENT, ret);
    mocker_clean();
}

void TcHostNotifyBaseAddrInit005()
{
    int ret;

    mocker(drvDeviceGetIndexByPhyId, 1, 0);
    mocker(halNotifyGetInfo, 1, 0);
    mocker(open, 1, 1);
    mocker(mmap, 1, 1);
    mocker(RsNotifyCfgSet, 1, 4);
    mocker(munmap, 1, 1);
    mocker(close, 1, 0);
    ret = HostNotifyBaseAddrInit(0);
    mocker_clean();
}

void TcHostNotifyBaseAddrInit006()
{
    int ret;

    mocker(drvDeviceGetIndexByPhyId, 1, 0);
    mocker(halNotifyGetInfo, 1, 0);
    mocker(open, 1, 1);
    mocker(mmap, 1, 1);
    mocker(RsNotifyCfgSet, 1, 4);
    mocker(munmap, 1, 0);
    mocker(close, 1, 0);
    ret = HostNotifyBaseAddrInit(0);
    mocker_clean();
    EXPECT_INT_EQ(4, ret);
}

void* StubMmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset)
{
    errno = 1;
    return (void*)-1;
};

void TcHostNotifyBaseAddrInit007()
{
    int ret;

    mocker_clean();
    mocker(drvDeviceGetIndexByPhyId, 1, 0);
    mocker(halNotifyGetInfo, 1, 0);
    mocker(open, 1, 1);
    mocker_invoke(mmap, StubMmap, 20);
    ret = HostNotifyBaseAddrInit(0);
    EXPECT_INT_EQ(-ENOMEM, ret);
    mocker_clean();
}

void TcHostNotifyBaseAddrUninit()
{
    int ret;

    mocker(RsNotifyCfgGet, 1, 0);
    mocker(open, 1, 0);
    mocker(ioctl, 1, 0);
    mocker(munmap, 1, 0);
    ret = HostNotifyBaseAddrUninit(0);
    mocker_clean();
    EXPECT_INT_NE(0, ret);
}

void TcHostNotifyBaseAddrUninit001()
{
    int ret;

    mocker(RsNotifyCfgGet, 1, 1);
    ret = HostNotifyBaseAddrUninit(0);
    mocker_clean();
    EXPECT_INT_EQ(1, ret);
}

void TcHostNotifyBaseAddrUninit002()
{
    int ret;

    mocker(RsNotifyCfgGet, 1, 0);
    mocker(open, 1, -1);
    mocker(drvDeviceGetIndexByPhyId, 1, 1);
    ret = HostNotifyBaseAddrUninit(0);
    mocker_clean();
    EXPECT_INT_EQ(-1, ret);
}

void TcHostNotifyBaseAddrUninit003()
{
    int ret;

    mocker(RsNotifyCfgGet, 1, 0);
    mocker(open, 1, 0);
    mocker(ioctl, 1, -1);
    ret = HostNotifyBaseAddrUninit(0);
    mocker_clean();
    EXPECT_INT_EQ(-ENOENT, ret);
}

void TcHostNotifyBaseAddrUninit004()
{
    int ret;

    mocker(RsNotifyCfgGet, 1, 0);
    mocker(open, 1, 0);
    mocker(ioctl, 1, 0);
    mocker(munmap, 1, 3);
    ret = HostNotifyBaseAddrUninit(0);
    mocker_clean();
    EXPECT_INT_EQ(-ENOENT, ret);
}

void TcHostNotifyBaseAddrUninit005()
{
    int ret;
    gNotifyFd = 1;
    mocker(RsNotifyCfgGet, 10, 0);
    mocker(open, 10, 1);
    mocker(ioctl, 10, 0);
    mocker(munmap, 1, 1);
    mocker(close, 1, 0);
    ret = HostNotifyBaseAddrUninit(0);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

void TcRaPeerSendWrlist()
{
    int ret;
    struct RaQpHandle qpHandle = {0};
    struct SendWrlistData wr = {0};
    struct SendWrRsp opRsp = {0};
    struct WrlistSendCompleteNum wrlistNum = {0};

    wrlistNum.sendNum = 1;
    mocker(RsSendWrlist, 1, 0);
    ret = RaPeerSendWrlist(&qpHandle, &wr, &opRsp, wrlistNum);
    mocker_clean();
}

void TcRaPeerSendWrlist001()
{
    int ret;
    struct RaQpHandle qpHandle = {0};
    struct SendWrlistData wr = {0};
    struct SendWrRsp opRsp = {0};
    struct WrlistSendCompleteNum wrlistNum = {0};

    wrlistNum.sendNum = 1;
    mocker(RsSendWrlist, 1, -1);
    ret = RaPeerSendWrlist(&qpHandle, &wr, &opRsp, wrlistNum);
    mocker_clean();
    EXPECT_INT_EQ(-1, ret);
}

void TcRaGetQpContext()
{
    struct RaQpHandle RaQpHandle = {0};
    void* qpHandle = (void*)&RaQpHandle;
    void* qp = NULL;
    void* sendCq = NULL;
    void* recvCq = NULL;
    struct RaRdmaOps ops = {0};
    RaQpHandle.rdmaOps = NULL;
    RaGetQpContext(qpHandle, &qp, &sendCq, &recvCq);
    RaGetQpContext(NULL, &qp, &sendCq, &recvCq);
    ops.raGetQpContext = RaPeerGetQpContext;
    RaQpHandle.rdmaOps = &ops;
    RaGetQpContext(qpHandle, &qp, &sendCq, &recvCq);
}

void TcRaCreateCq()
{
    struct ibv_cq* ibSendCq = NULL;
    struct ibv_cq* ibRecvCq = NULL;
    void* context = NULL;
    struct CqAttr attr = {0};
    attr.qpContext = &context;
    attr.ibSendCq = &ibSendCq;
    attr.ibRecvCq = &ibRecvCq;
    attr.sendCqDepth = 16384;
    attr.recvCqDepth = 16384;
    attr.sendCqEventId = 1;
    attr.recvCqEventId = 2;

    struct RaRdmaHandle RaRdmaHandle = {0};
    void* rdmaHandle = (void*)&RaRdmaHandle;
    RaRdmaHandle.rdevIndex = 0;
    RaRdmaHandle.rdevInfo.phyId = 32767;
    RaRdmaHandle.rdmaOps = NULL;
    RaCqCreate(rdmaHandle, &attr);
    RaCqDestroy(rdmaHandle, &attr);

    struct RaRdmaOps ops = {0};
    ops.raCqCreate = RaPeerCqCreate;
    ops.raCqDestroy = RaPeerCqDestroy;
    RaRdmaHandle.rdmaOps = &ops;
    RaCqCreate(rdmaHandle, &attr);
    RaCqDestroy(rdmaHandle, &attr);

    RaRdmaHandle.rdevInfo.phyId = 0;
    RaCqCreate(rdmaHandle, &attr);
    RaCqDestroy(rdmaHandle, &attr);

    mocker((stub_fn_t)RaPeerCqCreate, 1, 0);
    mocker((stub_fn_t)RaPeerCqDestroy, 1, 0);
    RaCqCreate(rdmaHandle, &attr);
    RaCqDestroy(rdmaHandle, &attr);
    mocker_clean();
}

void TcRaCreateNotmalQp()
{
    struct ibv_cq* ibSendCq = NULL;
    struct ibv_cq* ibRecvCq = NULL;
    void* context = NULL;
    struct ibv_qp_init_attr qpInitAttr = {0};
    qpInitAttr.qp_context = context;
    qpInitAttr.send_cq = ibSendCq;
    qpInitAttr.recv_cq = ibRecvCq;
    qpInitAttr.qp_type = 2;
    qpInitAttr.cap.max_inline_data = 32;
    qpInitAttr.cap.max_send_wr = 4096;
    qpInitAttr.cap.max_send_sge = 4096;
    qpInitAttr.cap.max_recv_wr = 4096;
    qpInitAttr.cap.max_recv_sge = 1;
    struct ibv_qp* qp;
    struct RaQpHandle RaQpHandle = {0};
    void* qpHandle = &RaQpHandle;

    struct RaRdmaHandle RaRdmaHandle = {0};
    void* rdmaHandle = (void*)&RaRdmaHandle;
    RaRdmaHandle.rdevIndex = 0;
    RaRdmaHandle.rdevInfo.phyId = 32767;
    RaRdmaHandle.rdmaOps = NULL;
    struct RaRdmaOps ops = {0};
    RaRdmaHandle.rdmaOps = &ops;
    ops.raNormalQpCreate = NULL;
    ops.raNormalQpDestroy = NULL;
    RaNormalQpCreate(rdmaHandle, &qpInitAttr, &qpHandle, (void**)&qp);
    RaQpHandle.rdmaOps = NULL;
    RaNormalQpDestroy(qpHandle);

    ops.raNormalQpCreate = RaPeerNormalQpCreate;
    ops.raNormalQpDestroy = RaPeerNormalQpDestroy;

    mocker((stub_fn_t)RaPeerNormalQpCreate, 10, 0);
    mocker((stub_fn_t)RaPeerNormalQpDestroy, 10, 0);
    RaNormalQpCreate(rdmaHandle, &qpInitAttr, &qpHandle, (void**)&qp);
    RaNormalQpDestroy(qpHandle);

    RaNormalQpCreate(rdmaHandle, &qpInitAttr, NULL, (void**)&qp);
    RaNormalQpDestroy(NULL);

    RaRdmaHandle.rdevInfo.phyId = 0;
    RaNormalQpCreate(rdmaHandle, &qpInitAttr, &qpHandle, (void**)&qp);
    RaNormalQpDestroy(qpHandle);
    mocker_clean();

    mocker((stub_fn_t)RaPeerNormalQpCreate, 10, -1);
    mocker((stub_fn_t)RaPeerNormalQpDestroy, 10, -1);
    RaNormalQpCreate(rdmaHandle, &qpInitAttr, &qpHandle, (void**)&qp);
    RaQpHandle.rdmaOps = &ops;
    RaNormalQpDestroy(qpHandle);
    mocker_clean();
}

void TcRaCreateCompChannel()
{
    struct RaRdmaHandle RaRdmaHandle = {0};
    void* rdmaHandle = (void*)&RaRdmaHandle;
    RaRdmaHandle.rdevIndex = 0;
    RaRdmaHandle.rdevInfo.phyId = 32767;
    RaRdmaHandle.rdmaOps = NULL;

    void* compChannel = NULL;
    RaCreateCompChannel(rdmaHandle, &compChannel);
    RaDestroyCompChannel(rdmaHandle, compChannel);

    compChannel = (void*)0xabcd;
    RaCreateCompChannel(rdmaHandle, &compChannel);
    RaDestroyCompChannel(rdmaHandle, compChannel);

    struct RaRdmaOps ops = {0};
    ops.raCreateCompChannel = RaPeerCreateCompChannel;
    ops.raDestroyCompChannel = RaPeerDestroyCompChannel;
    RaRdmaHandle.rdmaOps = &ops;
    RaCreateCompChannel(rdmaHandle, &compChannel);
    RaDestroyCompChannel(rdmaHandle, compChannel);

    RaCreateCompChannel(rdmaHandle, NULL);
    RaDestroyCompChannel(rdmaHandle, NULL);
    RaCreateCompChannel(NULL, NULL);
    RaDestroyCompChannel(NULL, NULL);

    RaRdmaHandle.rdevInfo.phyId = 0;
    RaCreateCompChannel(rdmaHandle, &compChannel);
    RaDestroyCompChannel(rdmaHandle, compChannel);
}

void TcRaGetCqeErrInfo()
{
    int ret;
    struct CqeErrInfo info = {0};

    ret = RaGetCqeErrInfo(0, NULL);
    EXPECT_INT_EQ(128103, ret);

    mocker(RaHdcGetCqeErrInfo, 1, 0);
    ret = RaGetCqeErrInfo(0, &info);
    EXPECT_INT_EQ(0, ret);

    ret = RaGetCqeErrInfo(128, &info);
    EXPECT_INT_NE(0, ret);
    return;
}

void TcRaRdevGetCqeErrInfoList()
{
    struct RaRdmaHandle raRdmaHandle = {0};
    struct CqeErrInfo info[128] = {{0}};
    unsigned int num = 128;
    int ret;

    raRdmaHandle.rdevIndex = 0;
    raRdmaHandle.rdevInfo.phyId = 32767;
    raRdmaHandle.rdmaOps = NULL;

    mocker(RaHdcGetCqeErrInfoList, 10, 0);
    ret = RaRdevGetCqeErrInfoList((void*)&raRdmaHandle, info, &num);
    EXPECT_INT_EQ(0, ret);

    ret = RaRdevGetCqeErrInfoList((void*)&raRdmaHandle, info, NULL);
    EXPECT_INT_EQ(128103, ret);

    num = 129;
    ret = RaRdevGetCqeErrInfoList((void*)&raRdmaHandle, info, &num);
    EXPECT_INT_EQ(128303, ret);
    mocker_clean();

    return;
}

void TcRaRsGetIfnum()
{
    int ret;
    int outLen = 0;
    int opResult = 0;
    int rcvBufLen = 300;

    char inBuf[512];
    char outBuf[512];

    ret = RaRsGetIfnum(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(0, ret);

    return;
}

void TcRaCreateSrq()
{
    struct RaRdmaHandle RaRdmaHandle = {0};
    void* rdmaHandle = (void*)&RaRdmaHandle;
    RaRdmaHandle.rdevIndex = 0;
    RaRdmaHandle.rdevInfo.phyId = 32767;
    RaRdmaHandle.rdmaOps = NULL;
    struct SrqAttr attr = {0};

    RaCreateSrq(rdmaHandle, NULL);
    RaDestroySrq(rdmaHandle, NULL);

    RaCreateSrq(rdmaHandle, &attr);
    RaDestroySrq(rdmaHandle, &attr);

    struct RaRdmaOps ops = {0};
    ops.raCreateSrq = RaPeerCreateSrq;
    ops.raDestroySrq = RaPeerDestroySrq;
    RaRdmaHandle.rdmaOps = &ops;
    RaCreateSrq(rdmaHandle, &attr);
    RaDestroySrq(rdmaHandle, &attr);

    RaCreateSrq(NULL, NULL);
    RaDestroySrq(NULL, NULL);

    RaRdmaHandle.rdevInfo.phyId = 0;
    RaCreateSrq(rdmaHandle, &attr);
    RaDestroySrq(rdmaHandle, &attr);
}

void TcRaRsSocketPortIsUse()
{
    unsigned int size = sizeof(union OpSocketConnectData) + sizeof(struct MsgHead);
    union OpSocketConnectData socketConnectData = {{0}};
    unsigned int port = 0x16;

    socketConnectData.txData.conn[0].port = port;
    socketConnectData.txData.num = 1;

    char* inBuf = calloc(1, size);
    char* outBuf = calloc(1, size);
    int outLen = 0;
    int opResult = 0;

    memcpy(inBuf + sizeof(struct MsgHead), &socketConnectData, sizeof(union OpSocketConnectData));
    memcpy(outBuf + sizeof(struct MsgHead), &socketConnectData, sizeof(union OpSocketConnectData));
    RaRsSocketBatchConnect(inBuf, outBuf, &outLen, &opResult, size);

    socketConnectData.txData.num = 1U | (1U << 31U);
    socketConnectData.txData.conn[0].port = 0xFFFFFFFF;
    memcpy(inBuf + sizeof(struct MsgHead), &socketConnectData, sizeof(union OpSocketConnectData));
    memcpy(outBuf + sizeof(struct MsgHead), &socketConnectData, sizeof(union OpSocketConnectData));
    RaRsSocketBatchConnect(inBuf, outBuf, &outLen, &opResult, size);

    free(inBuf);
    free(outBuf);
    inBuf = NULL;
    outBuf = NULL;

    size = sizeof(union OpSocketListenData) + sizeof(struct MsgHead);
    union OpSocketListenData socketListenData = {{0}};
    socketListenData.txData.conn[0].port = port;
    socketListenData.txData.num = 1;

    inBuf = calloc(1, size);
    outBuf = calloc(1, size);
    memcpy(inBuf + sizeof(struct MsgHead), &socketListenData, sizeof(union OpSocketListenData));
    memcpy(outBuf + sizeof(struct MsgHead), &socketListenData, sizeof(union OpSocketListenData));
    RaRsSocketListenStart(inBuf, outBuf, &outLen, &opResult, size);
    RaRsSocketListenStop(inBuf, outBuf, &outLen, &opResult, size);

    socketListenData.txData.num = 1U | (1U << 31U);
    socketListenData.txData.conn[0].port = 0xFFFFFFFF;
    memcpy(inBuf + sizeof(struct MsgHead), &socketListenData, sizeof(union OpSocketListenData));
    memcpy(outBuf + sizeof(struct MsgHead), &socketListenData, sizeof(union OpSocketListenData));
    RaRsSocketListenStart(inBuf, outBuf, &outLen, &opResult, size);
    RaRsSocketListenStop(inBuf, outBuf, &outLen, &opResult, size);

    free(inBuf);
    free(outBuf);
    inBuf = NULL;
    outBuf = NULL;
}

void TcRaRsGetVnicIpInfosV1()
{
    unsigned int size = sizeof(union OpGetVnicIpInfosDataV1) + sizeof(struct MsgHead);
    union OpGetVnicIpInfosDataV1 vnicInfos = {{0}};

    vnicInfos.txData.phyId = 0;
    vnicInfos.txData.type = 0;
    vnicInfos.txData.ids[0] = 3232235521;
    vnicInfos.txData.num = 1;

    char* inBuf = calloc(1, size);
    char* outBuf = calloc(1, size);
    int outLen = 0;
    int opResult = 0;

    memcpy(inBuf + sizeof(struct MsgHead), &vnicInfos, sizeof(union OpGetVnicIpInfosDataV1));
    memcpy(outBuf + sizeof(struct MsgHead), &vnicInfos, sizeof(union OpGetVnicIpInfosDataV1));
    RaRsGetVnicIpInfosV1(inBuf, outBuf, &outLen, &opResult, size);

    free(inBuf);
    free(outBuf);
    inBuf = NULL;
    outBuf = NULL;
}

void TcRaRsGetVnicIpInfos()
{
    unsigned int size = sizeof(union OpGetVnicIpInfosData) + sizeof(struct MsgHead);
    union OpGetVnicIpInfosData vnicInfos = {{0}};

    vnicInfos.txData.phyId = 0;
    vnicInfos.txData.type = 0;
    vnicInfos.txData.ids[0] = 3232235521;
    vnicInfos.txData.num = 1;

    char* inBuf = calloc(1, size);
    char* outBuf = calloc(1, size);
    int outLen = 0;
    int opResult = 0;

    memcpy(inBuf + sizeof(struct MsgHead), &vnicInfos, sizeof(union OpGetVnicIpInfosData));
    memcpy(outBuf + sizeof(struct MsgHead), &vnicInfos, sizeof(union OpGetVnicIpInfosData));
    RaRsGetVnicIpInfos(inBuf, outBuf, &outLen, &opResult, size);

    free(inBuf);
    free(outBuf);
    inBuf = NULL;
    outBuf = NULL;
}

void TcRaRsTypicalMrReg()
{
    int ret;
    int outLen = 0;
    int opResult = 0;
    int rcvBufLen = 300;

    char inBuf[512];
    char outBuf[512];

    ret = RaRsTypicalMrRegV1(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(ret, 0);
    ret = RaRsTypicalMrDereg(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(ret, 0);

    ret = RaRsTypicalMrReg(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(ret, 0);
    ret = RaRsTypicalMrDereg(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(ret, 0);
}

void TcRaRsTypicalQpCreate()
{
    int ret;
    int outLen = 0;
    int opResult = 0;
    int rcvBufLen = 300;

    char inBuf[512];
    char outBuf[512];

    ret = RaRsTypicalQpCreate(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(ret, 0);
    ret = RaRsTypicalQpModify(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(0, ret);
}

void TcRaHdcRecvHandleSendPktUnsuccess()
{
    mocker_clean();
    mocker(DlHalHdcRecv, 1, 1);
    mocker(DlDrvHdcAllocMsg, 1, 0);
    mocker(DlDrvHdcFreeMsg, 1, 1);
    mocker(DlDrvHdcSessionClose, 1, 1);
    mocker(RsSetCtx, 1, 0);
    mocker(pthread_mutex_lock, 1, 0);
    mocker(pthread_mutex_unlock, 1, 0);
    RaHdcRecvHandleSendPkt(0);
    mocker_clean();
}

void TcRaGetTlsEnable()
{
    struct RaInfo info = {0};
    bool tlsEnable = false;
    int ret;

    info.mode = NETWORK_PEER_ONLINE;
    ret = RaGetTlsEnable(&info, &tlsEnable);
    EXPECT_INT_EQ(0, ret);

    info.mode = NETWORK_OFFLINE;
    mocker(RaHdcProcessMsg, 1, 0);
    ret = RaGetTlsEnable(&info, &tlsEnable);
    EXPECT_INT_EQ(0, ret);

    info.phyId = RA_MAX_PHY_ID_NUM;
    ret = RaGetTlsEnable(&info, &tlsEnable);
    EXPECT_INT_EQ(128303, ret);
}

void TcRaGetSecRandom()
{
    struct RaInfo info = {0};
    unsigned int value = 0;
    int ret;

    mocker_clean();
    info.mode = NETWORK_PEER_ONLINE;
    ret = RaGetSecRandom(&info, NULL);
    EXPECT_INT_EQ(128303, ret);

    info.mode = NETWORK_OFFLINE;
    ret = RaGetSecRandom(&info, &value);
    EXPECT_INT_EQ(0, ret);

    info.mode = NETWORK_OFFLINE;
    mocker(RaHdcProcessMsg, 10, 0);
    mocker(RaPeerGetSecRandom, 10, -1);
    ret = RaGetSecRandom(&info, &value);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRaRsGetSecRandom()
{
    unsigned int size = sizeof(union OpGetSecRandomData) + sizeof(struct MsgHead);
    union OpGetSecRandomData opData = {{0}};

    char* inBuf = calloc(1, size);
    char* outBuf = calloc(1, size);
    int outLen = 0;
    int opResult = 0;
    int ret;

    memcpy(inBuf + sizeof(struct MsgHead), &opData, sizeof(union OpGetSecRandomData));
    memcpy(outBuf + sizeof(struct MsgHead), &opData, sizeof(union OpGetSecRandomData));
    ret = RaRsGetSecRandom(inBuf, outBuf, &outLen, &opResult, size);
    EXPECT_INT_EQ(0, ret);

    free(inBuf);
    free(outBuf);
    inBuf = NULL;
    outBuf = NULL;
}

void TcRaRsGetTlsEnable()
{
    unsigned int size = sizeof(union OpGetTlsEnableData) + sizeof(struct MsgHead);
    union OpGetTlsEnableData opData = {{0}};

    char* inBuf = calloc(1, size);
    char* outBuf = calloc(1, size);
    int outLen = 0;
    int opResult = 0;
    int ret;

    memcpy(inBuf + sizeof(struct MsgHead), &opData, sizeof(union OpGetTlsEnableData));
    memcpy(outBuf + sizeof(struct MsgHead), &opData, sizeof(union OpGetTlsEnableData));
    ret = RaRsGetTlsEnable(inBuf, outBuf, &outLen, &opResult, size);
    EXPECT_INT_EQ(0, ret);

    free(inBuf);
    free(outBuf);
    inBuf = NULL;
    outBuf = NULL;
}

static int StubRsGetHccnCfgValue(struct RaInfo* info, enum HccnCfgKey key, char* value, unsigned int* valueLen)
{
    const char cfgValue[] = "nslb_dp";

    (void)info;
    (void)key;
    if (value == NULL || valueLen == NULL || *valueLen < sizeof(cfgValue)) {
        return -EINVAL;
    }
    if (memcpy_s(value, *valueLen, cfgValue, sizeof(cfgValue)) != EOK) {
        return -EINVAL;
    }
    *valueLen = sizeof(cfgValue);
    return 0;
}

void TcRaGetHccnCfg()
{
    struct RaInfo info = {0};
    struct RaInitConfig cfg = {
        .phyId = 2,
        .nicPosition = NETWORK_PEER_ONLINE,
        .hdcType = 0,
    };
    char* value = calloc(1, 2048);
    unsigned int valLen = 2048;
    int ret;

    mocker_clean();
    info.mode = NETWORK_OFFLINE;
    ret = RaGetHccnCfg(NULL, HCCN_CFG_UDP_PORT_MODE, value, &valLen);
    EXPECT_INT_EQ(128303, ret);

    ret = RaGetHccnCfg(&info, HCCN_CFG_UDP_PORT_MODE, NULL, &valLen);
    EXPECT_INT_EQ(128303, ret);

    ret = RaGetHccnCfg(&info, HCCN_CFG_UDP_PORT_MODE, value, NULL);
    EXPECT_INT_EQ(128303, ret);

    valLen = 1024;
    ret = RaGetHccnCfg(&info, HCCN_CFG_UDP_PORT_MODE, value, &valLen);
    EXPECT_INT_EQ(128303, ret);

    info.phyId = 64;
    info.mode = NETWORK_OFFLINE;
    valLen = 2048;
    ret = RaGetHccnCfg(&info, HCCN_CFG_UDP_PORT_MODE, value, &valLen);
    EXPECT_INT_EQ(128303, ret);

    info.phyId = 0;
    mocker(RaHdcProcessMsg, 10, 0);
    ret = RaGetHccnCfg(&info, HCCN_CFG_UDP_PORT_MODE, value, &valLen);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    info.phyId = 2;
    info.mode = NETWORK_PEER_ONLINE;
    valLen = 2048;
    ret = RaPeerInit(&cfg, 1);
    EXPECT_INT_EQ(0, ret);

    mocker_invoke(RsGetHccnCfg, StubRsGetHccnCfgValue, 1);
    ret = RaGetHccnCfg(&info, HCCN_CFG_UDP_PORT_MODE, value, &valLen);
    EXPECT_INT_EQ(0, ret);
    EXPECT_STR_EQ("nslb_dp", value);
    EXPECT_INT_EQ(8, (int)valLen);
    mocker_clean();

    ret = RaPeerDeinit(&cfg);
    EXPECT_INT_EQ(0, ret);

    info.mode = NETWORK_ONLINE;
    valLen = 2048;
    ret = RaGetHccnCfg(&info, HCCN_CFG_UDP_PORT_MODE, value, &valLen);
    EXPECT_INT_EQ(528302, ret);

    free(value);
}

void TcRaRsGetHccnCfg()
{
    unsigned int size = sizeof(union OpGetHccnCfgData) + sizeof(struct MsgHead);
    union OpGetHccnCfgData opData = {{0}};

    char* inBuf = calloc(1, size);
    char* outBuf = calloc(1, size);
    int outLen = 0;
    int opResult = 0;
    int ret;

    memcpy(inBuf + sizeof(struct MsgHead), &opData, sizeof(union OpGetHccnCfgData));
    memcpy(outBuf + sizeof(struct MsgHead), &opData, sizeof(union OpGetHccnCfgData));
    ret = RaRsGetHccnCfg(inBuf, outBuf, &outLen, &opResult, size);
    EXPECT_INT_EQ(0, ret);

    free(inBuf);
    free(outBuf);
    inBuf = NULL;
    outBuf = NULL;
}

void TcRaSaveSnapshotInput()
{
    enum SaveSnapshotAction action = 0;
    struct RaInfo* info = NULL;
    int ret;

    ret = RaSaveSnapshot(info, action);
    EXPECT_INT_NE(0, ret);

    ret = RaRestoreSnapshot(info);
    EXPECT_INT_NE(0, ret);

    info = calloc(1, sizeof(struct RaInfo));
    info->phyId = RA_MAX_PHY_ID_NUM;
    info->mode = NETWORK_PEER_ONLINE;
    ret = RaSaveSnapshot(info, action);
    EXPECT_INT_EQ(0, ret);

    ret = RaRestoreSnapshot(info);
    EXPECT_INT_EQ(0, ret);

    info->phyId = 0;
    action = SAVE_SNAPSHOT_ACTION_POST_PROCESSING + 1;
    ret = RaSaveSnapshot(info, action);
    EXPECT_INT_NE(0, ret);

    info->mode = NETWORK_PEER_ONLINE;
    info->phyId = 0;
    ret = RaSaveSnapshot(info, SAVE_SNAPSHOT_ACTION_PRE_PROCESSING);
    EXPECT_INT_EQ(0, ret);

    ret = RaRestoreSnapshot(info);
    EXPECT_INT_EQ(0, ret);

    info->mode = NETWORK_OFFLINE + 1;
    ret = RaSaveSnapshot(info, SAVE_SNAPSHOT_ACTION_PRE_PROCESSING);
    EXPECT_INT_NE(0, ret);

    ret = RaRestoreSnapshot(info);
    EXPECT_INT_NE(0, ret);

    free(info);
    info = NULL;
}

void TcRaSaveSnapshotPre()
{
    struct RaInfo info = {0};
    struct RaRdmaHandle* rdmaHandle = NULL;
    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = 0;
    struct RdevInitInfo initInfo = {0};
    initInfo.disabledLiteThread = false;
    initInfo.mode = NETWORK_OFFLINE;
    initInfo.notifyType = NOTIFY;
    int ret;

    TcHdcEnvInit();

    mocker(RaRdevInitCheckIp, 10, 0);
    mocker((stub_fn_t)HdcSendRecvPkt, 10, 0);
    mocker_invoke(RaHdcGetInterfaceVersion, RaGetInterfaceVersionStub, 10);
    mocker_invoke(RaHdcGetLiteSupport, RaHdcGetLiteSupportStub, 10);
    mocker(RaHdcNotifyBaseAddrInit, 10, 0);
    gInterfaceVersion = 1;
    ret = RaRdevInitV2(initInfo, rdevInfo, (void**)&rdmaHandle);
    EXPECT_INT_EQ(ret, 0);

    info.mode = NETWORK_OFFLINE;
    rdmaHandle->supportLite = LITE_NOT_SUPPORT;
    ret = RaSaveSnapshot(&info, SAVE_SNAPSHOT_ACTION_PRE_PROCESSING);
    EXPECT_INT_EQ(0, ret);

    rdmaHandle->supportLite = LITE_ALIGN_4KB;
    rdmaHandle->threadStatus = LITE_THREAD_STATUS_RUNNING;
    ret = RaSaveSnapshot(&info, SAVE_SNAPSHOT_ACTION_PRE_PROCESSING);
    EXPECT_INT_EQ(128300, ret);

    ret = RaSaveSnapshot(&info, SAVE_SNAPSHOT_ACTION_PRE_PROCESSING);
    EXPECT_INT_EQ(128300, ret);

    ret = RaSaveSnapshot(&info, SAVE_SNAPSHOT_ACTION_POST_PROCESSING);
    EXPECT_INT_EQ(0, ret);

    ret = RaRdevDeinit(rdmaHandle, NOTIFY);
    TcHdcEnvDeinit();
}

void TcRaSaveSnapshotPost()
{
    struct RaInfo info = {0};
    struct RaRdmaHandle* rdmaHandle = NULL;
    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = 0;
    struct RdevInitInfo initInfo = {0};
    initInfo.disabledLiteThread = false;
    initInfo.mode = NETWORK_OFFLINE;
    initInfo.notifyType = NOTIFY;
    int ret;

    TcHdcEnvInit();

    mocker(RaRdevInitCheckIp, 10, 0);
    mocker((stub_fn_t)HdcSendRecvPkt, 10, 0);
    mocker_invoke(RaHdcGetInterfaceVersion, RaGetInterfaceVersionStub, 10);
    mocker_invoke(RaHdcGetLiteSupport, RaHdcGetLiteSupportStub, 10);
    mocker(RaHdcNotifyBaseAddrInit, 10, 0);
    gInterfaceVersion = 1;
    ret = RaRdevInitV2(initInfo, rdevInfo, (void**)&rdmaHandle);
    EXPECT_INT_EQ(0, ret);

    info.mode = NETWORK_OFFLINE;
    ret = RaRestoreSnapshot(&info);
    EXPECT_INT_EQ(128300, ret);

    rdmaHandle->supportLite = LITE_NOT_SUPPORT;
    ret = RaRestoreSnapshot(&info);
    EXPECT_INT_EQ(0, ret);

    rdmaHandle->threadStatus = LITE_THREAD_STATUS_RUNNING;
    rdmaHandle->supportLite = LITE_ALIGN_4KB;
    ret = RaSaveSnapshot(&info, SAVE_SNAPSHOT_ACTION_PRE_PROCESSING);
    EXPECT_INT_EQ(0, ret);

    ret = RaSaveSnapshot(&info, SAVE_SNAPSHOT_ACTION_PRE_PROCESSING);
    EXPECT_INT_NE(0, ret);

    ret = RaSaveSnapshot(&info, SAVE_SNAPSHOT_ACTION_PRE_PROCESSING);
    EXPECT_INT_NE(0, ret);

    ret = RaRestoreSnapshot(&info);
    EXPECT_INT_EQ(0, ret);

    ret = RaRestoreSnapshot(&info);
    EXPECT_INT_EQ(0, ret);

    ret = RaRdevDeinit(rdmaHandle, NOTIFY);
    TcHdcEnvDeinit();
}

void TcHdcAsyncDelReqHandle()
{
    pthread_mutex_t reqMutex;
    pthread_mutex_init(&reqMutex, NULL);

    struct RaListHead list1 = {0};

    RA_INIT_LIST_HEAD(&list1);
    RaHwAsyncDelList(&list1, &reqMutex);
}

void TcRaHdcDeinitAsyncAll() { RaHdcDeinitAsyncAll(); }
