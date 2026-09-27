/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ra_peer.h"
#include "ra_rs_err.h"
#include <errno.h>
#include <string.h>
#include "securec.h"
#include "ut_dispatch.h"
#include "rs.h"
#include "hccp.h"
extern int RaPeerSetConnParam(struct SocketInfoT conn[], struct SocketFdData rsConn[], unsigned int i, int bufSize);
extern int RaRdevInitCheckIp(int mode, struct rdev rdevInfo, char localIp[]);
extern int RaPeerLoopbackQpCreate(struct RaRdmaHandle* rdmaHandle, struct LoopbackQpPair* qpPair, void** qpHandle);
extern int
RaPeerLoopbackSingleQpCreate(struct RaRdmaHandle* rdmaHandle, struct RaQpHandle** qpHandle, struct ibv_qp** qp);
extern int RaPeerLoopbackQpModify(struct RaQpHandle* qpHandle0, struct RaQpHandle* qpHandle1);

int RaPeerLoopbackSingleQpCreateStub(struct RaRdmaHandle* rdmaHandle, struct RaQpHandle** qpHandle, struct ibv_qp** qp)
{
    static int callNum = 0;
    int ret = 0;
    callNum++;

    if (callNum == 1) {
        return RaPeerLoopbackSingleQpCreate(rdmaHandle, qpHandle, qp);
    } else {
        return -1;
    }
    return ret;
}

void TcPeer()
{
    int ret;
    int devId = 0;
    int flag = 0;
    int port = 0;
    int timeout = 100;
    void* addr = NULL;
    void* data = NULL;
    int size = 0;
    int maxSize = 2050;
    int access = 0;
    struct SendWr* wr = NULL;
    struct SendWrRsp wqeIndex = {0};
    int index = 0;
    unsigned long pa = 0;
    unsigned long long va = 0;
    struct QpPeerInfo* qpInfo = NULL;
    struct SocketConnectInfoT conn[1] = {{0}};
    struct SocketListenInfoT listen[1];
    struct SocketInfoT info[1] = {{0}};
    struct SocketCloseInfoT close[1] = {0};
    int sockFd = 1;
    void* qpHandle = NULL;
    void* qpHandleWithAttr = NULL;
    int status = 0;
    struct RaInitConfig config = {
        .phyId = devId,
        .nicPosition = 1,
        .hdcType = 0,
    };
    config.phyId = 0;
    int ipAddr = 0;
    unsigned int hostTgid = 0;
    int qpMode = 0;
    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = 0;
    struct RaRdmaHandle rdmaHandleTmp = {
        .rdevInfo = rdevInfo,
        .rdevIndex = 0,
    };
    struct RaSocketHandle socketHandleTmp = {
        .rdevInfo = rdevInfo,
    };
    struct RaRdmaHandle* rdmaHandle = &rdmaHandleTmp;
    struct RaSocketHandle* socketHandle = &socketHandleTmp;
    struct SocketFdData rsConn[] = {{0}};

    listen[0].socketHandle = socketHandle;
    conn[0].socketHandle = socketHandle;
    close[0].socketHandle = socketHandle;
    info[0].socketHandle = socketHandle;
    struct QpExtAttrs extAttrs = {0};
    extAttrs.version = QP_CREATE_WITH_ATTR_VERSION;
    extAttrs.qpMode = RA_RS_NOR_QP_MODE;

    ret = RaPeerInit(&config, 1);
    EXPECT_INT_EQ(0, ret);

    ret = RaPeerInit(&config, 1);
    EXPECT_INT_EQ(0, ret);

    ret = RaPeerSocketBatchConnect(0, conn, 1);
    EXPECT_INT_EQ(0, ret);

    ret = RaPeerSocketListenStart(0, listen, 1);

    ret = RaPeerSocketListenStop(0, listen, 1);

    ret = RaPeerGetSockets(0, 0, info, 1);
    EXPECT_INT_EQ(1, ret);

    info[0].socketHandle = socketHandle;
    info[0].status = 1;
    mocker((stub_fn_t)calloc, 10, NULL);
    ret = RaPeerGetSockets(0, 0, info, 1);
    EXPECT_INT_EQ(-12, ret);
    mocker_clean();

    info[0].fdHandle = calloc(1, sizeof(struct SocketPeerInfo));

    ret = RaPeerSocketSend(0, info[0].fdHandle, data, size);
    EXPECT_INT_EQ(0, size);

    ret = RaPeerSocketSend(0, info[0].fdHandle, data, maxSize);
    EXPECT_INT_EQ(maxSize, ret);

    ret = RaPeerSocketRecv(0, info[0].fdHandle, data, size);
    EXPECT_INT_EQ(0, size);

    ret = RaPeerSocketRecv(0, info[0].fdHandle, data, maxSize);
    EXPECT_INT_EQ(maxSize, ret);

    rsConn[0].phyId = 0;
    rsConn[0].fd = 0;
    ret = RaPeerSetConnParam(info, rsConn, 0, 0);
    EXPECT_INT_EQ(0, ret);

    unsigned int tempDepth = 128;
    unsigned int qpNum = 0;
    ret = RaPeerSetTsqpDepth(rdmaHandle, tempDepth, &qpNum);
    EXPECT_INT_EQ(0, ret);

    ret = RaPeerGetTsqpDepth(rdmaHandle, &tempDepth, &qpNum);
    EXPECT_INT_EQ(0, ret);

    ret = RaPeerQpCreate(rdmaHandle, flag, qpMode, &qpHandle);
    EXPECT_INT_EQ(0, ret);
    EXPECT_ADDR_NE(NULL, qpHandle);
    ret = RaPeerQpCreateWithAttrs(rdmaHandle, &extAttrs, &qpHandleWithAttr);
    EXPECT_INT_EQ(0, ret);
    EXPECT_ADDR_NE(NULL, qpHandleWithAttr);

    struct QosAttr QosAttr = {0};
    QosAttr.tc = 110;
    QosAttr.sl = 3;
    ret = RaPeerSetQpAttrQos(qpHandle, &QosAttr);
    EXPECT_INT_EQ(0, ret);

    unsigned int rdmaTimeout = 6;
    ret = RaPeerSetQpAttrTimeout(qpHandle, &rdmaTimeout);
    EXPECT_INT_EQ(0, ret);

    unsigned int retryCnt = 5;
    ret = RaPeerSetQpAttrRetryCnt(qpHandle, &retryCnt);
    EXPECT_INT_EQ(0, ret);

    ret = RaPeerNotifyBaseAddrInit(EVENTID, 0);
    EXPECT_INT_EQ(0, ret);

    ret = RaPeerNotifyBaseAddrInit(NO_USE, 0);
    EXPECT_INT_EQ(0, ret);

    ret = NotifyBaseAddrUninit(EVENTID, 0);
    EXPECT_INT_EQ(0, ret);

    ret = NotifyBaseAddrUninit(NO_USE, 0);
    EXPECT_INT_EQ(0, ret);

    ret = RaPeerQpConnectAsync(qpHandle, info[0].fdHandle);
    EXPECT_INT_EQ(0, size);

    close[0].fdHandle = info[0].fdHandle;
    ret = RaPeerSocketBatchClose(0, close, 1);
    EXPECT_INT_EQ(0, size);

    mocker(memset_s, 20, 1);
    RaPeerSocketBatchClose(0, close, 1);
    mocker_clean();

    ret = RaPeerGetQpStatus(qpHandle, &status);
    EXPECT_INT_EQ(0, ret);

    struct MrInfoT mrInfo = {0};
    mrInfo.addr = addr;
    mrInfo.size = size;
    mrInfo.access = access;

    void* mrHandle = NULL;

    ret = RaPeerMrReg(qpHandle, &mrInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RaPeerMrDereg(qpHandle, &mrInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RaPeerRegisterMr(rdmaHandle, &mrInfo, &mrHandle);
    EXPECT_INT_EQ(0, ret);

    ret = RaPeerDeregisterMr(rdmaHandle, mrHandle);
    EXPECT_INT_EQ(0, ret);

    void* compChannel = NULL;
    ret = RaPeerCreateCompChannel(rdmaHandle, &compChannel);
    EXPECT_INT_EQ(0, ret);

    ret = RaPeerDestroyCompChannel(compChannel);
    EXPECT_INT_EQ(0, ret);

    ret = RaPeerSendWr(qpHandle, wr, &wqeIndex);
    EXPECT_INT_EQ(0, ret);

    struct SrqAttr attr = {0};
    ret = RaPeerCreateSrq(rdmaHandle, &attr);
    EXPECT_INT_EQ(0, ret);

    ret = RaPeerDestroySrq(rdmaHandle, &attr);
    EXPECT_INT_EQ(0, ret);

    struct RecvWrlistData revWr = {0};
    revWr.wrId = 100;
    revWr.memList.lkey = 0xff;
    revWr.memList.addr = (uint64_t)addr;
    revWr.memList.len = size;
    unsigned int recvNum = 1;
    unsigned int revCompleteNum = 0;
    ret = RaPeerRecvWrlist(qpHandle, &revWr, recvNum, &revCompleteNum);
    EXPECT_INT_EQ(0, ret);

    unsigned long long notifySize = 0;
    ret = RaPeerGetNotifyBaseAddr(qpHandle, &va, &notifySize);
    EXPECT_INT_EQ(0, ret);

    ret = RaPeerQpDestroy(qpHandle);
    EXPECT_INT_EQ(0, ret);
    ret = RaPeerQpDestroy(qpHandleWithAttr);
    EXPECT_INT_EQ(0, ret);

    ret = RaPeerDeinit(&config);
    EXPECT_INT_EQ(0, ret);

    ret = RaPeerDeinit(&config);
    EXPECT_INT_EQ(0, ret);

    struct SocketConnectInfoT connectErrRs[1] = {0};
    connectErrRs[0].socketHandle = socketHandle;
    mocker((stub_fn_t)RsSocketBatchConnect, 10, -1);
    ret = RaPeerSocketBatchConnect(0, connectErrRs, 1);
    EXPECT_INT_EQ(-1, ret);
    mocker((stub_fn_t)RsSocketSetScopeId, 10, -2);
    ret = RaPeerSocketBatchConnect(0, connectErrRs, 1);
    EXPECT_INT_EQ(-2, ret);
    mocker_clean();

    struct SocketListenInfoT listenErrRs[1] = {0};
    listenErrRs[0].socketHandle = socketHandle;
    mocker((stub_fn_t)RsSocketListenStart, 10, -1);
    ret = RaPeerSocketListenStart(0, listenErrRs, 1);
    EXPECT_INT_NE(0, ret);
    mocker((stub_fn_t)RsSocketSetScopeId, 10, -2);
    ret = RaPeerSocketListenStart(0, listenErrRs, 1);
    EXPECT_INT_EQ(-2, ret);
    mocker_clean();

    struct SocketListenInfoT listenErrRs2[1];
    listenErrRs2[0].socketHandle = socketHandle;
    listenErrRs2[0].port = 0;
    mocker((stub_fn_t)RsSocketListenStop, 10, -1);
    ret = RaPeerSocketListenStop(0, listenErrRs2, 1);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    struct SocketInfoT infoErrRs[1];
    infoErrRs[0].socketHandle = socketHandle;
    infoErrRs[0].fdHandle = NULL;
    mocker((stub_fn_t)calloc, 10, NULL);
    ret = RaPeerGetSockets(0, 0, infoErrRs, 1);
    EXPECT_INT_EQ(1, ret);
    mocker_clean();

    mocker(RaPeerSetConnParam, 1, 1);
    ret = RaPeerGetSockets(0, 0, infoErrRs, 1);
    EXPECT_INT_EQ(1, ret);
    mocker_clean();

    struct SocketInfoT infoErrRs2[1];
    infoErrRs2[0].socketHandle = socketHandle;
    infoErrRs2[0].fdHandle = NULL;
    mocker((stub_fn_t)memcpy_s, 10, -1);
    ret = RaPeerGetSockets(0, 0, infoErrRs2, 1);
    EXPECT_INT_EQ(-ESAFEFUNC, ret);
    mocker_clean();

    struct SocketInfoT infoErrRs3[1];
    infoErrRs3[0].socketHandle = socketHandle;
    infoErrRs3[0].fdHandle = NULL;
    mocker_ret((stub_fn_t)memcpy_s, 0, 1, 1);
    ret = RaPeerGetSockets(0, 0, infoErrRs3, 1);
    EXPECT_INT_EQ(1, ret);
    mocker_clean();

    struct SocketInfoT infoErrRs4[1];
    infoErrRs4[0].socketHandle = socketHandle;
    infoErrRs4[0].fdHandle = NULL;
    mocker((stub_fn_t)RsGetSockets, 10, 0);
    mocker((stub_fn_t)RsGetSslEnable, 10, -1);
    ret = RaPeerGetSockets(0, 0, infoErrRs4, 1);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker((stub_fn_t)RsSetTsqpDepth, 10, -1);
    ret = RaPeerSetTsqpDepth(rdmaHandle, tempDepth, &qpNum);
    EXPECT_INT_EQ(-1, ret);

    mocker((stub_fn_t)RsGetTsqpDepth, 10, -1);
    ret = RaPeerGetTsqpDepth(rdmaHandle, &tempDepth, &qpNum);
    EXPECT_INT_EQ(-1, ret);

    qpHandle = NULL;
    qpHandleWithAttr = NULL;
    mocker((stub_fn_t)calloc, 10, NULL);
    ret = RaPeerQpCreate(rdmaHandle, flag, qpMode, &qpHandle);
    EXPECT_INT_EQ(-ENOMEM, ret);
    EXPECT_ADDR_EQ(NULL, qpHandle);
    ret = RaPeerQpCreateWithAttrs(rdmaHandle, &extAttrs, &qpHandleWithAttr);
    EXPECT_INT_EQ(-ENOMEM, ret);
    EXPECT_ADDR_EQ(NULL, qpHandleWithAttr);
    mocker_clean();

    mocker((stub_fn_t)RsQpCreate, 10, 1);
    mocker((stub_fn_t)RsQpCreateWithAttrs, 10, 1);
    ret = RaPeerQpCreate(rdmaHandle, flag, qpMode, &qpHandle);
    EXPECT_INT_EQ(1, ret);
    EXPECT_ADDR_EQ(NULL, qpHandle);
    ret = RaPeerQpCreateWithAttrs(rdmaHandle, &extAttrs, &qpHandleWithAttr);
    EXPECT_INT_EQ(1, ret);
    EXPECT_ADDR_EQ(NULL, qpHandleWithAttr);
    mocker_clean();

    ret = RaPeerQpCreate(rdmaHandle, flag, qpMode, &qpHandle);
    EXPECT_INT_EQ(0, ret);
    mocker((stub_fn_t)RsQpDestroy, 10, -1);
    ret = RaPeerQpDestroy(qpHandle);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    qpMode = 2;
    ret = RaPeerQpCreate(rdmaHandle, flag, qpMode, &qpHandle);
    EXPECT_INT_EQ(0, ret);

    ret = RaPeerSetQpAttrQos(qpHandle, &QosAttr);
    EXPECT_INT_EQ(0, ret);

    ret = RaPeerSetQpAttrTimeout(qpHandle, &timeout);
    EXPECT_INT_EQ(0, ret);

    ret = RaPeerSetQpAttrRetryCnt(qpHandle, &retryCnt);
    EXPECT_INT_EQ(0, ret);

    ret = RaPeerNotifyBaseAddrInit(1000, 0);
    EXPECT_INT_EQ(-EINVAL, ret);

    ret = NotifyBaseAddrUninit(1000, 0);
    EXPECT_INT_EQ(-EINVAL, ret);

    mocker((stub_fn_t)RsGetQpStatus, 10, -1);
    ret = RaPeerGetQpStatus(qpHandle, &status);
    EXPECT_INT_EQ(-1, ret);

    mocker((stub_fn_t)RsMrReg, 10, -1);
    ret = RaPeerMrReg(qpHandle, &mrInfo);
    EXPECT_INT_EQ(-1, ret);

    mocker((stub_fn_t)RsMrDereg, 10, -1);
    ret = RaPeerMrDereg(qpHandle, &mrInfo);
    EXPECT_INT_EQ(-1, ret);

    mocker((stub_fn_t)RsRegisterMr, 10, -1);
    ret = RaPeerRegisterMr(rdmaHandle, &mrInfo, &mrHandle);
    EXPECT_INT_EQ(-1, ret);

    mocker((stub_fn_t)RsDeregisterMr, 10, -1);
    ret = RaPeerDeregisterMr(rdmaHandle, mrHandle);
    EXPECT_INT_EQ(-1, ret);

    mocker((stub_fn_t)RsCreateCompChannel, 10, -1);
    ret = RaPeerCreateCompChannel(rdmaHandle, &compChannel);
    EXPECT_INT_EQ(-1, ret);

    mocker((stub_fn_t)RsDestroyCompChannel, 10, -1);
    ret = RaPeerDestroyCompChannel(compChannel);
    EXPECT_INT_EQ(-1, ret);

    struct SrqAttr attrSrq = {0};
    mocker((stub_fn_t)RsCreateSrq, 10, -1);
    ret = RaPeerCreateSrq(rdmaHandle, &attrSrq);
    EXPECT_INT_EQ(-1, ret);

    mocker((stub_fn_t)RsDestroySrq, 10, -1);
    ret = RaPeerDestroySrq(rdmaHandle, &attrSrq);
    EXPECT_INT_EQ(-1, ret);

    mocker((stub_fn_t)RsRecvWrlist, 10, -1);
    ret = RaPeerRecvWrlist(qpHandle, &revWr, recvNum, &revCompleteNum);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    struct SocketInfoT infoRs[1];
    infoRs[0].socketHandle = socketHandle;

    ret = RaPeerGetSockets(0, 0, infoRs, 1);
    EXPECT_INT_EQ(1, ret);

    infoRs[0].fdHandle = calloc(1, sizeof(struct SocketPeerInfo));

    mocker((stub_fn_t)RsQpConnectAsync, 10, -1);
    ret = RaPeerQpConnectAsync(qpHandle, infoRs[0].fdHandle);
    EXPECT_INT_EQ(-1, ret);

    mocker((stub_fn_t)RsGetNotifyMrInfo, 10, -1);
    ret = RaPeerGetNotifyBaseAddr(qpHandle, &va, &notifySize);
    EXPECT_INT_EQ(-1, ret);

    mocker((stub_fn_t)RsPeerSocketSend, 10, -1);
    ret = RaPeerSocketSend(devId, infoRs[0].fdHandle, data, size);
    EXPECT_INT_EQ(-1, ret);

    ret = RaPeerQpDestroy(qpHandle);
    EXPECT_INT_EQ(0, ret);

    struct SocketCloseInfoT closeRs[1] = {0};
    closeRs[0].fdHandle = infoRs[0].fdHandle;
    closeRs[0].socketHandle = socketHandle;
    mocker((stub_fn_t)RsSocketBatchClose, 10, -1);
    ret = RaPeerSocketBatchClose(0, closeRs, 1);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker((stub_fn_t)RsInit, 10, -1);
    ret = RaPeerInit(&config, 1);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker((stub_fn_t)RsDeinit, 10, -11);
    ret = RaPeerDeinit(&config);
    EXPECT_INT_EQ(-11, ret);
    mocker_clean();

    mocker((stub_fn_t)RsDeinit, 10, -1);
    ret = RaPeerDeinit(&config);
    EXPECT_INT_EQ(-1, ret);

    mocker_clean();

    return;
}

void TcPeerFail()
{
    struct RaSocketHandle socketHandle = {0};
    socketHandle.rdevInfo.phyId = 0;
    socketHandle.rdevInfo.family = 0;
    struct SocketConnectInfoT conn[1] = {{0}};
    conn[0].socketHandle = &socketHandle;
    conn[0].port = 0;
    struct SocketConnectInfo rsConn[1] = {0};
    mocker((stub_fn_t)memcpy_s, 10, -1);
    RaGetSocketConnectInfo(conn, 1, rsConn, 2);
    mocker_clean();

    struct SocketListenInfoT connListen[1];
    struct SocketListenInfo rsConnListen[1];
    connListen[0].phase = 0;
    connListen[0].err = 0;
    connListen[0].socketHandle = &socketHandle;
    connListen[0].port = 0;
    mocker((stub_fn_t)memcpy_s, 10, -1);
    RaGetSocketListenInfo(connListen, 1, rsConnListen, 2);
    mocker_clean();

    rsConnListen[0].phase = 0;
    rsConnListen[0].err = 0;
    rsConnListen[0].phyId = 0;
    rsConnListen[0].family = 0;
    connListen[0].socketHandle = &socketHandle;
    rsConnListen[0].port = 0;
    mocker((stub_fn_t)memcpy_s, 10, -1);
    RaGetSocketListenResult(rsConnListen, 1, connListen, 2);
    mocker_clean();

    struct SocketListenInfoT connListenInfo[1] = {0};
    connListenInfo[0].port = 0;
    connListenInfo[0].socketHandle = &socketHandle;
    mocker((stub_fn_t)RaGetSocketListenInfo, 10, 0);
    mocker((stub_fn_t)RsSocketListenStart, 10, -1);
    RaPeerSocketListenStart(0, connListenInfo, 1);
    mocker((stub_fn_t)RsSocketListenStop, 10, -1);
    RaPeerSocketListenStop(0, connListenInfo, 1);
    mocker_clean();

    struct SocketPeerInfo peerSocketHandle = {0};
    int ret;

    ret = RaPeerSocketSend(0, &peerSocketHandle, NULL, 0);
    EXPECT_INT_EQ(0, ret);

    peerSocketHandle.sslEnable = 1;
    ret = RaPeerSocketSend(0, &peerSocketHandle, NULL, 0);
    EXPECT_INT_EQ(0, ret);

    peerSocketHandle.sslEnable = 0;
    ret = RaPeerSocketRecv(0, &peerSocketHandle, NULL, 0);
    EXPECT_INT_EQ(0, ret);

    peerSocketHandle.sslEnable = 1;
    ret = RaPeerSocketRecv(0, &peerSocketHandle, NULL, 0);
    EXPECT_INT_EQ(0, ret);

    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 0;
    struct SocketWlistInfoT whiteList[1] = {{0}};
    mocker((stub_fn_t)inet_ntoa, 10, NULL);
    RaPeerSocketWhiteListAdd(rdevInfo, whiteList, 1);
    RaPeerSocketWhiteListDel(rdevInfo, whiteList, 1);
    mocker_clean();

    mocker((stub_fn_t)RsSocketDeinit, 10, -1);
    RaPeerSocketDeinit(rdevInfo);
    mocker_clean();

    mocker((stub_fn_t)RsPeerGetIfnum, 10, -1);
    unsigned int num = 0;
    RaPeerGetIfnum(0, &num);
    mocker_clean();

    struct InterfaceInfo interfaceInfos[1];
    mocker((stub_fn_t)RsPeerGetIfaddrs, 10, -1);
    RaPeerGetIfaddrs(0, interfaceInfos, &num);
    mocker_clean();

    return;
}

void TcRaPeerEpollCtlAdd()
{
    int ret;

    ret = RaPeerEpollCtlAdd(NULL, RA_EPOLLIN);
    EXPECT_INT_EQ(0, ret);

    mocker((stub_fn_t)RsEpollCtlAdd, 3, -1);
    ret = RaPeerEpollCtlAdd(NULL, RA_EPOLLIN);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    return;
}

void TcRaPeerSetTcpRecvCallback()
{
    RaSetTcpRecvCallback(NULL, NULL);
    (void)RaPeerSetTcpRecvCallback(0, NULL);

    struct RaSocketHandle abc = {0};
    int cb = 0;
    RaSetTcpRecvCallback(&abc, &cb);

    abc.rdevInfo.phyId = 10000;
    RaSetTcpRecvCallback(&abc, &cb);
    return;
}

void TcRaPeerEpollCtlMod()
{
    int ret;

    ret = RaPeerEpollCtlMod(NULL, RA_EPOLLIN);
    EXPECT_INT_EQ(0, ret);

    mocker((stub_fn_t)RsEpollCtlMod, 3, -1);
    ret = RaPeerEpollCtlMod(NULL, RA_EPOLLIN);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    return;
}

void TcRaPeerEpollCtlDel()
{
    int ret;
    struct SocketPeerInfo fdHandle = {0};

    ret = RaPeerEpollCtlDel((const void*)&fdHandle);
    EXPECT_INT_EQ(0, ret);

    mocker((stub_fn_t)RsEpollCtlDel, 3, -1);
    ret = RaPeerEpollCtlDel((const void*)&fdHandle);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    return;
}

void TcRaPeerCqCreate()
{
    int ret;
    struct RaRdmaHandle rdmaHandle = {0};
    rdmaHandle.rdevInfo.phyId = 0;
    rdmaHandle.rdevIndex = 0;

    struct ibv_cq* ibSendCq = NULL;
    struct ibv_cq* ibRecvCq = NULL;
    struct CqAttr attr = {0};
    attr.ibSendCq = &ibSendCq;
    attr.ibRecvCq = &ibRecvCq;
    attr.sendCqDepth = 16384;
    attr.recvCqDepth = 16384;
    attr.sendCqEventId = 1;
    attr.recvCqEventId = 2;

    ret = RaPeerCqCreate(&rdmaHandle, &attr);
    EXPECT_INT_EQ(0, ret);

    ret = RaPeerCqDestroy(&rdmaHandle, &attr);
    EXPECT_INT_EQ(0, ret);

    mocker((stub_fn_t)RsCqCreate, 3, -1);
    ret = RaPeerCqCreate(&rdmaHandle, &attr);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    ret = RaPeerCqCreate(&rdmaHandle, &attr);
    EXPECT_INT_EQ(0, ret);
    mocker((stub_fn_t)RsCqDestroy, 3, -1);
    ret = RaPeerCqDestroy(&rdmaHandle, &attr);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();
    return;
}

void TcRaPeerNormalQpCreate()
{
    int ret;
    struct RaQpHandle* qpHandle;
    struct ibv_qp_init_attr qpInitAttr = {0};
    struct RaRdmaHandle rdmaHandle = {0};
    rdmaHandle.rdevInfo.phyId = 0;
    rdmaHandle.rdevIndex = 0;
    void** qp = NULL;
    ret = RaPeerNormalQpCreate(&rdmaHandle, &qpInitAttr, (void**)&qpHandle, qp);
    EXPECT_INT_EQ(0, ret);

    ret = RaPeerNormalQpDestroy(qpHandle);
    EXPECT_INT_EQ(0, ret);

    mocker((stub_fn_t)calloc, 3, NULL);
    ret = RaPeerNormalQpCreate(&rdmaHandle, &qpInitAttr, (void**)&qpHandle, qp);
    EXPECT_INT_EQ(-ENOMEM, ret);
    mocker_clean();

    mocker((stub_fn_t)RsNormalQpCreate, 3, -1);
    ret = RaPeerNormalQpCreate(&rdmaHandle, &qpInitAttr, (void**)&qpHandle, qp);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    ret = RaPeerNormalQpCreate(&rdmaHandle, &qpInitAttr, (void**)&qpHandle, qp);
    EXPECT_INT_EQ(0, ret);
    mocker((stub_fn_t)RsNormalQpDestroy, 3, -1);
    ret = RaPeerNormalQpDestroy(qpHandle);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();
    return;
}

void TcRaPeerCreateEventHandle()
{
    int ret;
    int fd;

    ret = RaPeerCreateEventHandle(&fd);
    EXPECT_INT_EQ(0, ret);
}

void TcRaPeerCtlEventHandle()
{
    int ret;
    int fdHandle;

    ret = RaPeerCtlEventHandle(0, NULL, 0, RA_EPOLLONESHOT);
    EXPECT_INT_EQ(-EINVAL, ret);

    ret = RaPeerCtlEventHandle(0, &fdHandle, 1, RA_EPOLLONESHOT);
    EXPECT_INT_EQ(0, ret);
}

void TcRaPeerWaitEventHandle()
{
    int ret;
    int fd;

    ret = RaPeerWaitEventHandle(0, NULL, 0, -1, 0);
    EXPECT_INT_EQ(0, ret);
}

void TcRaPeerDestroyEventHandle()
{
    int ret;
    int fd;

    ret = RaPeerDestroyEventHandle(&fd);
    EXPECT_INT_EQ(0, ret);
}

void TcRaLoopbackQpCreate()
{
    struct RaRdmaHandle* rdmaHandle2;
    struct RaRdmaHandle* rdmaHandle;
    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = 0;
    int ret = 0;

    mocker(RaRdevInitCheckIp, 10, 0);
    ret = RaRdevInit(NETWORK_PEER_ONLINE, NO_USE, rdevInfo, (void**)&rdmaHandle);
    EXPECT_INT_EQ(0, ret);
    ret = RaRdevGetHandle(rdevInfo.phyId, (void**)&rdmaHandle2);
    EXPECT_INT_EQ(0, ret);
    EXPECT_INT_EQ((intptr_t)rdmaHandle, (intptr_t)rdmaHandle2);
    mocker_clean();

    struct LoopbackQpPair qpPair = {0};
    void* qpHandle = NULL;

    ret = RaLoopbackQpCreate(NULL, NULL, NULL);
    EXPECT_INT_EQ(128103, ret);

    ret = RaLoopbackQpCreate(rdmaHandle, NULL, NULL);
    EXPECT_INT_EQ(128103, ret);

    ret = RaLoopbackQpCreate(rdmaHandle, &qpPair, NULL);
    EXPECT_INT_EQ(128103, ret);

    rdmaHandle->rdevInfo.phyId = 128;
    ret = RaLoopbackQpCreate(rdmaHandle, &qpPair, &qpHandle);
    EXPECT_INT_EQ(128103, ret);

    rdmaHandle->rdevInfo.phyId = 0;
    mocker(RaPeerLoopbackQpCreate, 10, -1);
    ret = RaLoopbackQpCreate(rdmaHandle, &qpPair, &qpHandle);
    EXPECT_INT_EQ(128100, ret);

    mocker_clean();
    ret = RaLoopbackQpCreate(rdmaHandle, &qpPair, &qpHandle);
    EXPECT_INT_EQ(0, ret);
    ret = RaQpDestroy(qpHandle);
    EXPECT_INT_EQ(0, ret);

    ret = RaRdevDeinit(rdmaHandle, NO_USE);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRaPeerLoopbackQpCreate()
{
    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = 0;
    struct RaRdmaHandle rdmaHandleTmp = {
        .rdevInfo = rdevInfo,
        .rdevIndex = 0,
    };
    struct LoopbackQpPair qpPair = {0};
    void* qpHandle = NULL;
    int ret = 0;

    mocker(RaPeerLoopbackSingleQpCreate, 10, -1);
    ret = RaPeerLoopbackQpCreate(&rdmaHandleTmp, &qpPair, &qpHandle);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker_invoke(RaPeerLoopbackSingleQpCreate, RaPeerLoopbackSingleQpCreateStub, 10);
    ret = RaPeerLoopbackQpCreate(&rdmaHandleTmp, &qpPair, &qpHandle);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker(RaPeerLoopbackQpModify, 10, -1);
    ret = RaPeerLoopbackQpCreate(&rdmaHandleTmp, &qpPair, &qpHandle);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();
}

void TcRaPeerLoopbackSingleQpCreate()
{
    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = 0;
    struct RaRdmaHandle rdmaHandleTmp = {
        .rdevInfo = rdevInfo,
        .rdevIndex = 0,
    };
    struct RaQpHandle* qpHandle = NULL;
    struct ibv_qp* qp = NULL;
    int ret = 0;

    mocker(RaPeerCqCreate, 10, -1);
    ret = RaPeerLoopbackSingleQpCreate(&rdmaHandleTmp, &qpHandle, &qp);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker(RaPeerNormalQpCreate, 10, -1);
    ret = RaPeerLoopbackSingleQpCreate(&rdmaHandleTmp, &qpHandle, &qp);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();
}

void TcRaPeerSetQpLbValue()
{
    struct RaQpHandle qpHandle = {0};
    int lbValue = 0;
    int ret = 0;

    mocker_clean();
    mocker(RsSetQpLbValue, 10, -1);
    ret = RaPeerSetQpLbValue(&qpHandle, lbValue);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();
}

void TcRaPeerGetQpLbValue()
{
    struct RaQpHandle qpHandle = {0};
    int lbValue = 0;
    int ret = 0;

    mocker_clean();
    mocker(RsGetQpLbValue, 10, -1);
    ret = RaPeerGetQpLbValue(&qpHandle, &lbValue);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();
}

void TcRaPeerGetLbMax()
{
    struct RaRdmaHandle rdmaHandle = {0};
    int lbMax = 0;
    int ret = 0;

    mocker_clean();
    mocker(RsGetLbMax, 10, -1);
    ret = RaPeerGetLbMax(&rdmaHandle, &lbMax);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();
}

int gRaPeerRsCapturedMode = -1;
unsigned int gRaPeerRsCapturedPhyId = 0;
int gRaPeerRsCapturedKey = -1;

int StubRsGetHccnCfgCaptureInfo(struct RaInfo* info, enum HccnCfgKey key, char* value, unsigned int* valueLen)
{
    const char* cfgValue = "nslb_dp";

    gRaPeerRsCapturedMode = (int)info->mode;
    gRaPeerRsCapturedPhyId = info->phyId;
    gRaPeerRsCapturedKey = (int)key;
    if (key == HCCN_CFG_QOS_DSCP) {
        cfgValue = "0:33,1:65";
    }
    if (memcpy_s(value, *valueLen, cfgValue, strlen(cfgValue)) != EOK) {
        return -EINVAL;
    }
    *valueLen = (unsigned int)(strlen(cfgValue) + 1);
    return 0;
}

void TcRaPeerGetHccnCfg()
{
    unsigned int phyId = 3;
    struct RaInitConfig config = {
        .phyId = phyId,
        .nicPosition = NETWORK_PEER_ONLINE,
        .hdcType = 0,
    };
    struct RaInfo info = {
        .mode = NETWORK_PEER_ONLINE,
        .phyId = phyId,
    };
    char value[2048] = {0};
    unsigned int valueLen = 2048;
    int ret = 0;

    ret = RaPeerInit(&config, 1);
    EXPECT_INT_EQ(0, ret);

    ret = RaPeerGetHccnCfg(&info, HCCN_CFG_UDP_PORT_MODE, value, &valueLen);
    EXPECT_INT_EQ(0, ret);

    mocker_invoke(RsGetHccnCfg, StubRsGetHccnCfgCaptureInfo, 2);
    ret = RaPeerGetHccnCfg(&info, HCCN_CFG_QOS_DSCP, value, &valueLen);
    EXPECT_INT_EQ(0, ret);
    EXPECT_INT_EQ(NETWORK_PEER_ONLINE, gRaPeerRsCapturedMode);
    EXPECT_INT_EQ((int)phyId, (int)gRaPeerRsCapturedPhyId);
    EXPECT_INT_EQ(HCCN_CFG_QOS_DSCP, gRaPeerRsCapturedKey);
    EXPECT_STR_EQ("0:33,1:65", value);
    EXPECT_INT_EQ(10, (int)valueLen);

    memset_s(value, sizeof(value), 0, sizeof(value));
    valueLen = 2048;
    ret = RaPeerGetHccnCfg(&info, HCCN_CFG_UDP_PORT_MODE, value, &valueLen);
    EXPECT_INT_EQ(0, ret);
    EXPECT_INT_EQ(NETWORK_PEER_ONLINE, gRaPeerRsCapturedMode);
    EXPECT_INT_EQ((int)phyId, (int)gRaPeerRsCapturedPhyId);
    EXPECT_INT_EQ(HCCN_CFG_UDP_PORT_MODE, gRaPeerRsCapturedKey);
    EXPECT_STR_EQ("nslb_dp", value);
    EXPECT_INT_EQ(8, (int)valueLen);
    mocker_clean();

    ret = RaPeerDeinit(&config);
    EXPECT_INT_EQ(0, ret);
}
