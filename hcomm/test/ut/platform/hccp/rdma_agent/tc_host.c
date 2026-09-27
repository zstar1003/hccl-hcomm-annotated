/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccp.h"
#include "securec.h"
#include "ut_dispatch.h"
#include "ra_client_host.h"
#include "ra_hdc.h"
#include "ra_hdc_rdma_notify.h"
#include "ra_hdc_rdma.h"
#include "ra_hdc_socket.h"
#include "ra_peer.h"
#include "config_log.h"

extern struct RaSocketOps gRaPeerSocketOps;
extern int HdcSendRecvPkt(void* session, void* pSendRcvBuf, unsigned int inBufLen, unsigned int outDataLen);
extern int RaInetPton(int family, union HccpIpAddr ip, char netAddr[], unsigned int len);
extern int RaHdcNotifyBaseAddrInit(unsigned int notifyType, unsigned int phyId, unsigned long long** notifyVa);
extern int RaRdevInitCheck(int mode, struct rdev rdevInfo, char localIp[], unsigned int num, void* rdmaHandle);
extern int RaRdevInitCheckIp(int mode, struct rdev rdevInfo, char localIp[]);
extern void RsSetCtx(unsigned int phyId);
extern int
RsSocketGetClientSocketErrInfo(struct SocketConnectInfo conn[], struct SocketErrInfo err[], unsigned int num);
extern int
RsSocketGetServerSocketErrInfo(struct SocketListenInfo conn[], struct ServerSocketErrInfo err[], unsigned int num);
extern int RsSocketAcceptCreditAdd(struct SocketListenInfo conn[], uint32_t num, unsigned int creditLimit);
extern int RaPeerSocketAcceptCreditAdd(
    unsigned int phyId, struct SocketListenInfoT conn[], unsigned int num, unsigned int creditLimit);

extern int
RaIfaddrInfoConverter(unsigned int phyId, bool isAll, struct InterfaceInfo interfaceInfos[], unsigned int* num);
extern int DlHalInit(void);
extern void DlHalDeinit(void);

extern unsigned int gInterfaceVersion;

int RaHdcGetIfaddrsStub(unsigned int phyId, struct IfaddrInfo ifaddrInfos[], unsigned int* num)
{
    *num = 4;
    return 0;
}

int RaGetInterfaceVersionStub(unsigned int phyId, unsigned int interfaceOpcode, unsigned int* interfaceVersion)
{
    *interfaceVersion = gInterfaceVersion;
    return 0;
}

void TcIfaddr()
{
    int ret;

    mocker_invoke(RaHdcGetInterfaceVersion, RaGetInterfaceVersionStub, 100);
    gInterfaceVersion = 0;
    unsigned int ifaddrNum = 4;
    struct InterfaceInfo interfaceInfos[4] = {0};
    bool isAll = false;

    ret = RaIfaddrInfoConverter(0, isAll, interfaceInfos, &ifaddrNum);
    EXPECT_INT_EQ(-22, ret);

    gInterfaceVersion = 1;
    ifaddrNum = 4;
    ret = RaIfaddrInfoConverter(0, isAll, interfaceInfos, &ifaddrNum);
    EXPECT_INT_EQ(-22, ret);

    mocker(RaHdcGetIfaddrs, 100, 0);
    gInterfaceVersion = 1;
    ifaddrNum = 4;
    ret = RaIfaddrInfoConverter(0, isAll, interfaceInfos, &ifaddrNum);
    EXPECT_INT_EQ(0, ret);

    gInterfaceVersion = 2;
    ifaddrNum = 4;
    ret = RaIfaddrInfoConverter(0, isAll, interfaceInfos, &ifaddrNum);
    EXPECT_INT_EQ(-22, ret);

    gInterfaceVersion = 3;
    ifaddrNum = 4;
    ret = RaIfaddrInfoConverter(0, isAll, interfaceInfos, &ifaddrNum);
    EXPECT_INT_EQ(-22, ret);

    gInterfaceVersion = 1;
    ifaddrNum = 4;
    mocker(calloc, 10, 0);
    ret = RaIfaddrInfoConverter(0, isAll, interfaceInfos, &ifaddrNum);
    EXPECT_INT_EQ(-22, ret);
    mocker_clean();

    return;
}

int StubRaHdcSendWrV2(struct RaQpHandle* qpHdc, struct SendWrV2* wr, struct SendWrRsp* opRsp) { return 0; }

int RaGetQpHyperFeatureStub(struct RaQpHandle* qpStub, struct HyperFeature* hyperFeature) { return 0; }

void TcHost()
{
    DlHalInit();
    int ret = 0;
    struct RdevInitInfo initInfo = {0};
    struct rdev rdevInfo = {0};
    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = 0;
    struct MrInfoT mrInfo = {0};
    mocker((stub_fn_t)HdcSendRecvPkt, 200, 0);
    mocker_invoke(RaHdcGetInterfaceVersion, RaGetInterfaceVersionStub, 200);
    gInterfaceVersion = 2;

    ret = RaRdevInitWithBackup(NULL, NULL, NULL, NULL);
    EXPECT_INT_NE(0, ret);

    struct RaRdmaHandle* rdmaHandleBakcup = NULL;
    ret = RaRdevInitWithBackup(&initInfo, &rdevInfo, &rdevInfo, (void**)&rdmaHandleBakcup);
    EXPECT_INT_NE(0, ret);

    RaRdevDeinit(NULL, NOTIFY);
    struct RaRdmaHandle* rdmaHandle = NULL;
    struct RaRdmaHandle* rdmaHandle2 = NULL;
    ret = RaRdevInit(NETWORK_OFFLINE, NOTIFY, rdevInfo, NULL);
    EXPECT_INT_NE(0, ret);

    ret = RaRdevInit(5, NOTIFY, rdevInfo, (void**)&rdmaHandle);
    EXPECT_INT_NE(0, ret);

    ret = RaRdevInit(NETWORK_OFFLINE, NOTIFY, rdevInfo, (void**)&rdmaHandle);
    EXPECT_INT_EQ(328008, ret);

    mocker(RaRdevInitCheckIp, 10, 0);
    ret = RaRdevInit(NETWORK_OFFLINE, NOTIFY, rdevInfo, (void**)&rdmaHandle);
    EXPECT_INT_EQ(0, ret);
    ret = RaRdevGetHandle(rdevInfo.phyId, (void**)&rdmaHandle2);
    EXPECT_INT_EQ(0, ret);
    EXPECT_ADDR_EQ(rdmaHandle, rdmaHandle2);
    mocker_clean();

    mocker((stub_fn_t)HdcSendRecvPkt, 200, 0);
    mocker_invoke(RaHdcGetInterfaceVersion, RaGetInterfaceVersionStub, 200);
    struct RaSocketHandle* socketHandle = NULL;
    ret = RaSocketInit(NETWORK_OFFLINE, rdevInfo, (void**)&socketHandle);
    EXPECT_INT_EQ(0, ret);
    ret = RaQpCreate(NULL, 0, 1, NULL);
    EXPECT_INT_NE(0, ret);
    ret = RaQpCreateWithAttrs(NULL, NULL, NULL);
    EXPECT_INT_NE(0, ret);
    ret = RaAiQpCreate(NULL, NULL, NULL, NULL);
    EXPECT_INT_NE(0, ret);

    RaGetIfnum(NULL, NULL);

    int ifnum = 0;
    struct RaInitConfig config = {0};
    struct RaGetIfattr ifattrConfig = {0};
    ifattrConfig.nicPosition = NETWORK_PEER_ONLINE;
    ret = RaGetIfnum(&ifattrConfig, &ifnum);
    EXPECT_INT_EQ(0, ret);

    ifattrConfig.phyId = 0;
    ifattrConfig.nicPosition = NETWORK_OFFLINE;
    ret = RaGetIfnum(&ifattrConfig, &ifnum);
    EXPECT_INT_EQ(0, ret);

    ifattrConfig.isAll = true;
    ret = RaGetIfnum(&ifattrConfig, &ifnum);
    EXPECT_INT_NE(0, ret);
    ifattrConfig.isAll = false;

    ifattrConfig.nicPosition = 5;
    ret = RaGetIfnum(&ifattrConfig, &ifnum);
    EXPECT_INT_EQ(228304, ret);

    ifattrConfig.phyId = 129;
    ifattrConfig.nicPosition = NETWORK_OFFLINE;
    ret = RaGetIfnum(&ifattrConfig, &ifnum);
    EXPECT_INT_EQ(128303, ret);

    ifattrConfig.nicPosition = 5;
    ret = RaGetIfnum(&ifattrConfig, &ifnum);
    EXPECT_INT_EQ(128303, ret);

    RaGetIfaddrs(NULL, NULL, NULL);
    RaSocketWhiteListAdd(socketHandle, NULL, 0);
    RaSocketWhiteListDel(socketHandle, NULL, 0);

    ret = RaSocketSetWhiteListStatus(0);
    EXPECT_INT_EQ(0, ret);
    ret = RaSocketSetWhiteListStatus(3);
    EXPECT_INT_EQ(128203, ret);
    ret = RaSocketGetWhiteListStatus(NULL);
    EXPECT_INT_EQ(128203, ret);
    unsigned int enable;
    ret = RaSocketGetWhiteListStatus(&enable);
    EXPECT_INT_EQ(0, ret);

    RaSocketListenStart(NULL, 0);
    RaSocketListenStop(NULL, 0);
    RaGetSockets(0, NULL, 0, NULL);
    RaSocketSend(NULL, NULL, 0, NULL);
    RaSocketRecv(NULL, NULL, 0, NULL);
    RaGetQpStatus(NULL, NULL);
    RaMrReg(NULL, NULL);
    RaMrDereg(NULL, NULL);
    RaSendWr(NULL, NULL, NULL);
    RaSendWrlist(NULL, NULL, NULL, 0, NULL);
    RaGetNotifyBaseAddr(NULL, NULL, NULL);
    RaGetNotifyMrInfo(NULL, NULL);
    RaQpDestroy(NULL);
    RaQpConnectAsync(NULL, NULL);
    RaRegisterMr(NULL, NULL, NULL);
    RaDeregisterMr(NULL, NULL);
    RaGetCqeErrInfo(0, NULL);
    RaGetQpAttr(NULL, NULL);

    ret = RaSendWrV2(NULL, NULL, NULL);
    EXPECT_INT_NE(0, ret);
    struct RaQpHandle raQpHandleV2 = {0};
    struct SendWrV2 wrV2 = {0};
    struct SendWrRsp opRspV2 = {0};
    struct SgList listV2 = {0};
    listV2.len = 0xFFFFFFFF;
    wrV2.bufList = &listV2;
    ret = RaSendWrV2(&raQpHandleV2, &wrV2, &opRspV2);
    EXPECT_INT_NE(0, ret);

    listV2.len = 0x1;
    ret = RaSendWrV2(&raQpHandleV2, &wrV2, &opRspV2);
    EXPECT_INT_NE(0, ret);

    struct RaRdmaOps rdmaOpsV2 = {0};
    rdmaOpsV2.raSendWrV2 = StubRaHdcSendWrV2;
    raQpHandleV2.rdmaOps = &rdmaOpsV2;
    ret = RaSendWrV2(&raQpHandleV2, &wrV2, &opRspV2);
    EXPECT_INT_EQ(0, ret);

    int devid = 0;
    int raCaseOther = 2;
    unsigned int remoteIp[1] = {0};
    struct SocketWlistInfoT whiteList[1] = {{0}};
    int flag = 0;
    int port = 0;
    int timeout = 0;
    void* addr = malloc(1);
    void* data = malloc(1);
    unsigned long long size = 1;
    int access = 1;

    struct SendWr* wr = calloc(1, sizeof(struct SendWr));
    struct SgList list[2];
    list[0].len = 1;
    list[1].len = 2147483649;
    int wqeIndex = 0;
    wr->bufList = list;
    int index = 0;
    unsigned long pa = 1;
    unsigned long long va = 1;
    int sockFd = 1;
    struct RaRdmaOps rdmaOps = {0};

    struct SocketHdcInfo* hdcSocketHandle = calloc(1, sizeof(struct SocketHdcInfo));
    struct SocketConnectInfoT conn[1] = {{0}};
    conn[0].socketHandle = NULL;
    RaSocketBatchConnect(conn, 1);
    RaSocketBatchClose((struct SocketCloseInfoT*)conn, 1);

    conn[0].socketHandle = socketHandle;
    struct SocketInfoT connTmp = {0};
    struct SocketCloseInfoT close[1] = {0};
    struct SocketListenInfoT listen[1];

    listen[0].socketHandle = NULL;
    RaSocketListenStart(listen, 1);
    RaSocketListenStop(listen, 1);

    listen[0].socketHandle = socketHandle;
    close[0].socketHandle = socketHandle;
    close[0].fdHandle = hdcSocketHandle;

    struct SocketInfoT socketInfo[1] = {0};

    struct RaInitConfig offlineConfig = {
        .phyId = 0,
        .nicPosition = 1,
        .hdcType = HDC_SERVICE_TYPE_RDMA,
    };
    struct RaGetIfattr offlineIfattrConfig = {
        .phyId = 0,
        .nicPosition = 1,
        .isAll = 0,
    };
    int qpStatus = 0;
    int server = 0;
    int client = 1;
    struct SendWrRsp opRsp = {0};

    struct SendWrlistData wrlistSend[1];
    struct SendWrRsp wrlistRsp[1];
    unsigned int ifaddrNum = 4;
    struct InterfaceInfo interfaceInfos[4] = {0};
    ret = RaGetIfaddrs(&offlineIfattrConfig, interfaceInfos, &ifaddrNum);
    EXPECT_INT_EQ(0, ret);

    ifaddrNum = 0;
    ret = RaGetIfaddrs(&offlineIfattrConfig, interfaceInfos, &ifaddrNum);
    EXPECT_INT_EQ(128303, ret);

    ifaddrNum = 9;
    ret = RaGetIfaddrs(&offlineIfattrConfig, interfaceInfos, &ifaddrNum);
    EXPECT_INT_EQ(128303, ret);

    ifaddrNum = 4;
    offlineIfattrConfig.phyId = 128;
    ret = RaGetIfaddrs(&offlineIfattrConfig, interfaceInfos, &ifaddrNum);
    EXPECT_INT_EQ(128303, ret);

    offlineIfattrConfig.phyId = 0;
    offlineIfattrConfig.nicPosition = NETWORK_PEER_ONLINE;
    ret = RaGetIfaddrs(&offlineIfattrConfig, interfaceInfos, &ifaddrNum);
    EXPECT_INT_EQ(0, ret);

    offlineIfattrConfig.nicPosition = 5;
    ret = RaGetIfaddrs(&offlineIfattrConfig, interfaceInfos, &ifaddrNum);
    EXPECT_INT_EQ(228304, ret);

    offlineIfattrConfig.isAll = true;
    ret = RaGetIfaddrs(&offlineIfattrConfig, interfaceInfos, &ifaddrNum);
    EXPECT_INT_NE(0, ret);
    offlineIfattrConfig.isAll = false;

    mocker_invoke(RaHdcGetIfaddrs, RaHdcGetIfaddrsStub, 10);
    ifaddrNum = 4;
    ret = RaIfaddrInfoConverter(0, 0, interfaceInfos, &ifaddrNum);
    EXPECT_INT_EQ(0, ret);

    mocker_clean();
    mocker((stub_fn_t)HdcSendRecvPkt, 200, 0);
    offlineConfig.nicPosition = 1;
    unsigned int interfaceVersion = 0;
    ret = RaGetInterfaceVersion(0, 0, &interfaceVersion);
    EXPECT_INT_EQ(0, ret);

    interfaceVersion = 0;
    ret = RaGetInterfaceVersion(0, 12, &interfaceVersion);
    EXPECT_INT_EQ(0, ret);

    interfaceVersion = 0;
    ret = RaGetInterfaceVersion(0, 24, &interfaceVersion);
    EXPECT_INT_EQ(0, ret);

    interfaceVersion = 0;
    ret = RaGetInterfaceVersion(0, 25, NULL);
    EXPECT_INT_EQ(128303, ret);

    ret = ConverReturnCode(0, 100001);
    EXPECT_INT_EQ(100001, ret);

    mocker_invoke(RaHdcGetInterfaceVersion, RaGetInterfaceVersionStub, 200);
    gInterfaceVersion = 2;
    ret = RaSocketWhiteListAdd(socketHandle, whiteList, 1);
    EXPECT_INT_EQ(0, ret);
    ret = RaSocketWhiteListDel(socketHandle, whiteList, 1);
    EXPECT_INT_EQ(0, ret);

    ret = RaSocketBatchConnect(NULL, 1);
    EXPECT_INT_NE(0, ret);
    ret = RaSocketBatchConnect(conn, 1);
    EXPECT_INT_EQ(0, ret);

    unsigned long long sentSize = 0;
    ret = RaSocketSend(hdcSocketHandle, data, 1, &sentSize);
    EXPECT_INT_EQ(128203, ret);

    unsigned long long receivedSize = 0;
    ret = RaSocketRecv(hdcSocketHandle, data, 1, &receivedSize);
    EXPECT_INT_EQ(128203, ret);

    ret = RaSocketBatchClose(NULL, 1);
    EXPECT_INT_NE(0, ret);
    ret = RaSocketBatchClose(close, 1);
    EXPECT_INT_EQ(0, ret);

    hdcSocketHandle = calloc(1, sizeof(struct SocketHdcInfo));

    socketInfo[0].socketHandle = NULL;
    unsigned int connectedNum = 0;
    RaGetSockets(0, socketInfo, 1, &connectedNum);

    socketInfo[0].fdHandle = hdcSocketHandle;
    socketInfo[0].socketHandle = socketHandle;
    ret = RaGetSockets(0, socketInfo, 1, &connectedNum);
    EXPECT_INT_EQ(0, ret);

    ret = RaSocketListenStart(listen, 1);
    EXPECT_INT_EQ(0, ret);

    socketHandle->rdevInfo.family = AF_INET;
    socketHandle->rdevInfo.phyId = 0;
    ret = RaSocketListenStop(listen, 1);
    EXPECT_INT_EQ(0, ret);

    socketHandle->rdevInfo.phyId = 8;
    ret = RaSocketSend(hdcSocketHandle, data, 1, &sentSize);

    ret = RaSocketRecv(hdcSocketHandle, data, 1, &receivedSize);

    mocker(RaHdcSocketSend, 10, 1);
    ret = RaSocketSend(hdcSocketHandle, data, 1, &sentSize);

    mocker(RaHdcSocketRecv, 10, 1);
    ret = RaSocketRecv(hdcSocketHandle, data, 1, &receivedSize);

    socketHandle->rdevInfo.phyId = 129;
    ret = RaSocketWhiteListAdd(socketHandle, whiteList, 1);
    EXPECT_INT_NE(0, ret);
    ret = RaSocketWhiteListDel(socketHandle, whiteList, 1);
    EXPECT_INT_NE(0, ret);

    ret = RaSocketBatchConnect(conn, 1);
    EXPECT_INT_NE(0, ret);

    ret = RaSocketSend(hdcSocketHandle, data, 1, &sentSize);
    EXPECT_INT_NE(0, ret);

    ret = RaSocketRecv(hdcSocketHandle, data, 1, &receivedSize);
    EXPECT_INT_NE(0, ret);

    ret = RaSocketBatchClose(close, 1);
    EXPECT_INT_NE(0, ret);

    RaGetSockets(0, socketInfo, 1, &connectedNum);
    EXPECT_INT_NE(0, ret);

    ret = RaSocketListenStart(listen, 1);
    EXPECT_INT_NE(0, ret);
    socketHandle->rdevInfo.family = AF_INET;
    ret = RaSocketListenStop(listen, 1);
    EXPECT_INT_NE(0, ret);

    ret = RaQpCreate(rdmaHandle, 3, 0, NULL);
    EXPECT_INT_NE(0, ret);
    ret = RaQpCreateWithAttrs(rdmaHandle, NULL, NULL);
    EXPECT_INT_NE(0, ret);
    ret = RaAiQpCreate(rdmaHandle, NULL, NULL, NULL);
    EXPECT_INT_NE(0, ret);
    struct RaQpHandle* qpHandle = NULL;
    struct RaQpHandle* qpHandleWithAttr = NULL;
    struct RaQpHandle* typicalQpHandle = NULL;
    struct RaQpHandle* aiQpHandle = NULL;
    struct AiQpInfo info = {0};
    ret = RaQpCreateWithAttrs(rdmaHandle, NULL, (void**)&qpHandleWithAttr);
    EXPECT_INT_NE(0, ret);
    ret = RaAiQpCreate(rdmaHandle, NULL, &info, (void**)&aiQpHandle);
    EXPECT_INT_NE(0, ret);
    struct QpExtAttrs extAttrs = {0};
    extAttrs.version = 0;
    ret = RaQpCreateWithAttrs(rdmaHandle, &extAttrs, (void**)&qpHandleWithAttr);
    EXPECT_INT_NE(0, ret);
    ret = RaAiQpCreate(rdmaHandle, &extAttrs, &info, (void**)&aiQpHandle);
    EXPECT_INT_NE(0, ret);
    extAttrs.version = QP_CREATE_WITH_ATTR_VERSION;
    extAttrs.qpMode = -1;
    ret = RaQpCreateWithAttrs(rdmaHandle, &extAttrs, (void**)&qpHandleWithAttr);
    EXPECT_INT_NE(0, ret);
    ret = RaAiQpCreate(rdmaHandle, &extAttrs, &info, (void**)&qpHandleWithAttr);
    EXPECT_INT_NE(0, ret);
    extAttrs.qpMode = RA_RS_GDR_TMPL_QP_MODE;
    ret = RaQpCreate(rdmaHandle, 1, 0, (void**)&qpHandle);
    EXPECT_INT_NE(0, ret);
    ret = RaQpCreate(rdmaHandle, 0, 0, (void**)&qpHandle);
    EXPECT_INT_EQ(0, ret);
    ret = RaQpCreateWithAttrs(rdmaHandle, &extAttrs, (void**)&qpHandleWithAttr);
    EXPECT_INT_EQ(0, ret);
    ret = RaAiQpCreate(rdmaHandle, &extAttrs, &info, (void**)&aiQpHandle);
    EXPECT_INT_EQ(0, ret);
    ret = RaQpDestroy(qpHandleWithAttr);
    EXPECT_INT_EQ(0, ret);
    ret = RaQpDestroy(aiQpHandle);
    EXPECT_INT_EQ(0, ret);

    ret = RaQpBatchModify(NULL, NULL, 0, 0);
    EXPECT_INT_NE(0, ret);

    struct RaQpHandle* batchModifyQpHdc[1];
    batchModifyQpHdc[0] = qpHandle;

    ret = RaQpBatchModify(rdmaHandle, (void**)batchModifyQpHdc, 1, 5);
    EXPECT_INT_EQ(0, ret);

    ret = RaQpBatchModify(rdmaHandle, (void**)batchModifyQpHdc, 1, 1);
    EXPECT_INT_EQ(0, ret);

    struct RaRdmaHandle rdmaHandleErr = {0};
    rdmaHandleErr = *rdmaHandle;
    (&rdmaHandleErr)->rdevInfo.phyId = 128;

    ret = RaQpBatchModify(&rdmaHandleErr, (void**)batchModifyQpHdc, 1, 1);
    EXPECT_INT_NE(0, ret);

    batchModifyQpHdc[0] = NULL;
    ret = RaQpBatchModify(rdmaHandle, (void**)batchModifyQpHdc, 1, 5);
    EXPECT_INT_NE(0, ret);

    unsigned int tempDepth, qpNum;
    ret = RaGetTsqpDepth(NULL, &tempDepth, &qpNum);
    EXPECT_INT_EQ(128103, ret);

    ret = RaGetTsqpDepth(rdmaHandle, NULL, &qpNum);
    EXPECT_INT_EQ(128103, ret);

    ret = RaGetTsqpDepth(rdmaHandle, &tempDepth, &qpNum);
    EXPECT_INT_EQ(0, ret);

    ret = RaSetTsqpDepth(NULL, tempDepth, &qpNum);
    EXPECT_INT_EQ(128103, ret);

    ret = RaSetTsqpDepth(rdmaHandle, tempDepth, NULL);
    EXPECT_INT_EQ(128103, ret);

    tempDepth = 1;
    ret = RaSetTsqpDepth(rdmaHandle, tempDepth, &qpNum);
    EXPECT_INT_EQ(128103, ret);

    tempDepth = 8;
    ret = RaSetTsqpDepth(rdmaHandle, tempDepth, &qpNum);
    EXPECT_INT_EQ(0, ret);

    struct RaQpHandle* qpHandleWrlist = NULL;
    ret = RaQpCreate(rdmaHandle, 0, 0, (void**)&qpHandleWrlist);
    EXPECT_INT_EQ(0, ret);
    qpHandleWrlist->rdmaOps = NULL;
    unsigned int completeNum = 1;
    wrlistSend[0].memList.len = 1;
    ret = RaSendWrlist(qpHandleWrlist, wrlistSend, wrlistRsp, 1, &completeNum);
    qpHandleWrlist->rdmaOps = rdmaHandle->rdmaOps;
    ret = RaQpDestroy(qpHandleWrlist);
    EXPECT_INT_EQ(0, ret);

    mrInfo.addr = addr;
    mrInfo.access = access;
    mrInfo.lkey = 0;
    mrInfo.size = size;
    ret = RaMrReg(qpHandle, &mrInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RaGetNotifyBaseAddr(rdmaHandle, &va, &size);
    EXPECT_INT_EQ(0, ret);
    struct MrInfoT mrInfo2 = {0};
    ret = RaGetNotifyMrInfo(rdmaHandle, &mrInfo2);
    EXPECT_INT_EQ(0, ret);

    ret = RaGetQpStatus(qpHandle, &qpStatus);
    EXPECT_INT_EQ(0, ret);

    ret = RaSendWr(qpHandle, wr, &opRsp);
    EXPECT_INT_EQ(0, ret);

    ret = RaQpConnectAsync(qpHandle, hdcSocketHandle);
    EXPECT_INT_EQ(0, ret);

    ret = RaMrDereg(qpHandle, &mrInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RaQpDestroy(qpHandle);
    EXPECT_INT_EQ(0, ret);

    struct TypicalQp qpInfo = {0};
    ret = RaTypicalQpCreate(NULL, 0, RA_RS_OP_QP_MODE_EXT, &qpInfo, (void**)&typicalQpHandle);
    EXPECT_INT_NE(0, ret);
    ret = RaTypicalQpCreate(rdmaHandle, 0, RA_RS_OP_QP_MODE_EXT, &qpInfo, NULL);
    EXPECT_INT_NE(0, ret);
    ret = RaTypicalQpCreate(rdmaHandle, 0, RA_RS_OP_QP_MODE_EXT, NULL, (void**)&typicalQpHandle);
    EXPECT_INT_NE(0, ret);
    ret = RaTypicalQpCreate(rdmaHandle, 3, RA_RS_OP_QP_MODE_EXT, &qpInfo, (void**)&typicalQpHandle);
    EXPECT_INT_NE(0, ret);
    ret = RaTypicalQpCreate(rdmaHandle, 0, -1, &qpInfo, (void**)&typicalQpHandle);
    EXPECT_INT_NE(0, ret);
    ret = RaTypicalQpCreate(rdmaHandle, 0, RA_RS_OP_QP_MODE_EXT, &qpInfo, (void**)&typicalQpHandle);
    EXPECT_INT_EQ(0, ret);

    struct TypicalQp remoteQpInfo = {0};
    ret = RaTypicalQpModify(typicalQpHandle, NULL, &remoteQpInfo);
    EXPECT_INT_NE(0, ret);
    ret = RaTypicalQpModify(NULL, &qpInfo, &remoteQpInfo);
    EXPECT_INT_NE(0, ret);
    ret = RaTypicalQpModify(typicalQpHandle, &qpInfo, &remoteQpInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RaTypicalSendWr(NULL, wr, &opRsp);
    EXPECT_INT_NE(0, ret);
    ret = RaTypicalSendWr(typicalQpHandle, wr, &opRsp);
    EXPECT_INT_EQ(0, ret);

    ret = RaQpDestroy(typicalQpHandle);
    EXPECT_INT_EQ(0, ret);

    struct RaQpHandle* qpHandle1 = NULL;
    struct RaQpHandle* qpHandle1WithAttr = NULL;
    struct RaQpHandle* typicalQpHandle1 = NULL;
    ret = RaQpCreate(rdmaHandle, 0, 5, (void**)&qpHandle1);
    EXPECT_INT_NE(0, ret);

    struct QpAttr qpAttrForErr = {0};
    ret = RaGetQpAttr(qpHandle1, &qpAttrForErr);
    EXPECT_INT_NE(0, ret);

    struct QpAttr attr = {0};
    struct RaQpHandle qpHandleTmp = {0};
    struct RaRdmaOps qpHyperFeatureOps = {0};
    qpHandleTmp.rdmaHandle = rdmaHandle;
    qpHyperFeatureOps.raGetQpHyperFeature = RaGetQpHyperFeatureStub;
    qpHandleTmp.rdmaOps = &qpHyperFeatureOps;
    ret = RaGetQpAttr(&qpHandleTmp, &attr);
    EXPECT_INT_EQ(0, ret);

    rdmaHandle->rdevInfo.phyId = 129;
    ret = RaQpCreate(rdmaHandle, 0, 0, (void**)&qpHandle1);
    EXPECT_INT_NE(0, ret);
    ret = RaQpCreateWithAttrs(rdmaHandle, &extAttrs, (void**)&qpHandle1WithAttr);
    EXPECT_INT_NE(0, ret);
    ret = RaTypicalQpCreate(rdmaHandle, 0, RA_RS_OP_QP_MODE_EXT, &qpInfo, (void**)&typicalQpHandle1);
    EXPECT_INT_NE(0, ret);

    ret = RaGetTsqpDepth(rdmaHandle, &tempDepth, &qpNum);
    EXPECT_INT_EQ(128103, ret);
    ret = RaSetTsqpDepth(rdmaHandle, tempDepth, &qpNum);
    EXPECT_INT_EQ(128103, ret);

    /* Save global ops pointers before NULL-ing to avoid polluting gRaHdcRdmaOps across iterations */
    void* savedRaQpCreate = (void*)(uintptr_t)rdmaHandle->rdmaOps->raQpCreate;
    void* savedRaQpCreateWithAttrs = (void*)(uintptr_t)rdmaHandle->rdmaOps->raQpCreateWithAttrs;
    void* savedRaTypicalQpCreate = (void*)(uintptr_t)rdmaHandle->rdmaOps->raTypicalQpCreate;
    rdmaHandle->rdmaOps->raQpCreate = NULL;
    rdmaHandle->rdmaOps->raQpCreateWithAttrs = NULL;
    rdmaHandle->rdmaOps->raTypicalQpCreate = NULL;
    ret = RaQpCreate(rdmaHandle, 0, 0, (void**)&qpHandle1);
    EXPECT_INT_NE(0, ret);
    ret = RaQpCreateWithAttrs(rdmaHandle, &extAttrs, (void**)&qpHandle1WithAttr);
    EXPECT_INT_NE(0, ret);
    ret = RaTypicalQpCreate(rdmaHandle, 0, RA_RS_OP_QP_MODE_EXT, &qpInfo, (void**)&typicalQpHandle1);
    EXPECT_INT_NE(0, ret);
    /* Restore global ops pointers so subsequent iterations are not affected */
    rdmaHandle->rdmaOps->raQpCreate = savedRaQpCreate;
    rdmaHandle->rdmaOps->raQpCreateWithAttrs = savedRaQpCreateWithAttrs;
    rdmaHandle->rdmaOps->raTypicalQpCreate = savedRaTypicalQpCreate;

    struct RaQpHandle qpHandle2 = {0};
    qpHandle = &qpHandle2;
    qpHandle->rdmaOps = NULL;
    mrInfo.addr = addr;
    mrInfo.access = access;
    mrInfo.lkey = 0;
    mrInfo.size = size;
    ret = RaMrReg(qpHandle, &mrInfo);
    EXPECT_INT_NE(0, ret);

    rdmaHandle->rdevInfo.phyId = 0;
    ret = RaGetNotifyBaseAddr(rdmaHandle, &va, &size);
    EXPECT_INT_EQ(0, ret);
    ret = RaGetNotifyMrInfo(rdmaHandle, &mrInfo2);
    EXPECT_INT_EQ(0, ret);

    ret = RaGetQpStatus(qpHandle, &qpStatus);
    EXPECT_INT_NE(0, ret);

    ret = RaSendWr(qpHandle, wr, &opRsp);
    EXPECT_INT_NE(0, ret);

    ret = RaQpConnectAsync(qpHandle, hdcSocketHandle);
    EXPECT_INT_NE(0, ret);

    ret = RaMrDereg(qpHandle, &mrInfo);
    EXPECT_INT_NE(0, ret);

    ret = RaQpDestroy(qpHandle);
    EXPECT_INT_NE(0, ret);

    socketHandle->rdevInfo.phyId = 128;
    ret = RaSocketDeinit(socketHandle);
    EXPECT_INT_EQ(128003, ret);

    mocker(RaHdcSocketDeinit, 1, -1);
    socketHandle->rdevInfo.phyId = 0;
    ret = RaSocketDeinit(socketHandle);
    EXPECT_INT_EQ(128000, ret);

    rdmaHandle->rdevInfo.phyId = 128;
    ret = RaRdevDeinit(rdmaHandle, NOTIFY);
    EXPECT_INT_EQ(128003, ret);

    rdmaHandle->rdevInfo.phyId = 0;
    ret = RaRdevDeinit(rdmaHandle, NOTIFY);
    EXPECT_INT_EQ(0, ret);

    mocker_clean();

    ret = RaInit(NULL);
    EXPECT_INT_EQ(128003, ret);
    ret = RaDeinit(NULL);
    EXPECT_INT_EQ(128003, ret);
    struct RaInitConfig erroDeviceConfig = {
        .phyId = 129,
        .nicPosition = 0,
        .hdcType = HDC_SERVICE_TYPE_RDMA,
    };
    ret = RaInit(&erroDeviceConfig);
    EXPECT_INT_EQ(128003, ret);

    ret = RaDeinit(&erroDeviceConfig);
    EXPECT_INT_EQ(128003, ret);

    struct RaInitConfig deviceConfig = {
        .phyId = 0,
        .nicPosition = 0,
        .hdcType = 0,
    };
    ret = RaInit(&deviceConfig);
    EXPECT_INT_EQ(0, ret);

    mocker(RaHdcInit, 10, 0);
    ret = RaInit(&offlineConfig);
    mocker_clean();

    mocker(RaHdcInit, 10, -1);
    ret = RaInit(&offlineConfig);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    mocker_invoke(RaHdcGetInterfaceVersion, RaGetInterfaceVersionStub, 10);
    ret = RaGetInterfaceVersion(0, 24, &interfaceVersion);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    struct RaInitConfig onlineConfig = {
        .phyId = 0,
        .nicPosition = 2,
        .hdcType = HDC_SERVICE_TYPE_RDMA,
    };
    ret = RaInit(&onlineConfig);
    EXPECT_INT_EQ(228004, ret);

    mocker(drvGetProcessSign, 10, -1);
    ret = RaInit(&offlineConfig);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    mocker(strcpy_s, 10, -1);
    ret = RaInit(&offlineConfig);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    ret = RaDeinit(&deviceConfig);
    EXPECT_INT_EQ(0, ret);

    deviceConfig.nicPosition = 5;
    ret = RaDeinit(&deviceConfig);
    EXPECT_INT_NE(0, ret);

    mocker(RaHdcDeinit, 10, 0);
    ret = RaDeinit(&offlineConfig);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    mocker(RaHdcDeinit, 10, -1);
    ret = RaDeinit(&offlineConfig);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    mocker(RaInetPton, 10, 0);
    ret = RaSocketInit(NETWORK_OFFLINE, rdevInfo, (void**)&socketHandle);
    mocker_clean();

    mocker(RaInetPton, 10, -22);
    ret = RaSocketInit(NETWORK_OFFLINE, rdevInfo, (void**)&socketHandle);
    mocker_clean();

    mocker(memcpy_s, 10, -1);
    mocker_invoke(RaHdcGetInterfaceVersion, RaGetInterfaceVersionStub, 10);
    ret = RaSocketInit(NETWORK_OFFLINE, rdevInfo, (void**)&socketHandle);
    EXPECT_INT_NE(0, ret);

    ret = RaRdevInit(NETWORK_OFFLINE, NOTIFY, rdevInfo, (void**)&rdmaHandle);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    mocker(calloc, 10, 0);
    mocker_invoke(RaHdcGetInterfaceVersion, RaGetInterfaceVersionStub, 10);
    ret = RaSocketInit(NETWORK_OFFLINE, rdevInfo, (void**)&socketHandle);
    EXPECT_INT_NE(0, ret);

    ret = RaRdevInit(NETWORK_OFFLINE, NOTIFY, rdevInfo, (void**)&rdmaHandle);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    mocker_invoke(RaHdcGetInterfaceVersion, RaGetInterfaceVersionStub, 100);
    rdevInfo.phyId = 129;
    ret = RaSocketInit(NETWORK_OFFLINE, rdevInfo, (void**)&socketHandle);
    EXPECT_INT_NE(0, ret);

    ret = RaRdevInit(NETWORK_OFFLINE, NOTIFY, rdevInfo, (void**)&rdmaHandle);
    EXPECT_INT_NE(0, ret);

    ret = RaSocketDeinit(NULL);
    EXPECT_INT_NE(0, ret);

    rdevInfo.phyId = 0;
    ret = RaSocketInit(5, rdevInfo, (void**)&socketHandle);
    EXPECT_INT_NE(0, ret);

    ret = RaRdevInit(5, NOTIFY, rdevInfo, (void**)&rdmaHandle);
    EXPECT_INT_NE(0, ret);

    mocker(RaRdevInitCheck, 2, 0);
    mocker(RaHdcNotifyBaseAddrInit, 5, 0);
    mocker(calloc, 10, 0);
    ret = RaRdevInit(5, NOTIFY, rdevInfo, (void**)&rdmaHandle);
    EXPECT_INT_EQ(328000, ret);
    mocker_clean();

    rdevInfo.phyId = 0;
    rdevInfo.family = 0;
    rdevInfo.localIp.addr.s_addr = 0;
    char localIp[1];
    mocker(RaGetIfaddrs, 10, 0);
    mocker(RaInetPton, 10, -1);
    RaRdevInitCheckIp(NETWORK_OFFLINE, rdevInfo, localIp);
    mocker_clean();

    rdevInfo.family = AF_INET;
    mocker(RaRdevInitCheck, 2, 0);
    mocker(RaHdcNotifyBaseAddrInit, 5, 0);
    ret = RaRdevInit(10, NOTIFY, rdevInfo, (void**)&rdmaHandle);
    EXPECT_INT_EQ(128003, ret);
    mocker_clean();

    mocker(RaRdevInitCheck, 2, 0);
    mocker(RaPeerNotifyBaseAddrInit, 5, 0);
    mocker(memcpy_s, 10, -1);
    ret = RaRdevInit(NETWORK_PEER_ONLINE, NOTIFY, rdevInfo, (void**)&rdmaHandle);
    EXPECT_INT_EQ(328006, ret);
    mocker_clean();

    mocker(RaRdevInitCheck, 2, 0);
    mocker(RaHdcNotifyBaseAddrInit, 5, 0);
    ret = RaRdevInit(NETWORK_OFFLINE, NOTIFY, rdevInfo, (void**)&rdmaHandle);
    EXPECT_INT_EQ(128003, ret);
    mocker_clean();

    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = 7;
    mocker(RaGetIfaddrs, 10, 0);
    ret = RaRdevInit(NETWORK_OFFLINE, NOTIFY, rdevInfo, (void**)&rdmaHandle);
    EXPECT_INT_EQ(328008, ret);
    mocker_clean();

    unsigned long long* notifyVa = NULL;
    mocker(drvDeviceGetIndexByPhyId, 10, 1);
    ret = RaHdcNotifyBaseAddrInit(NOTIFY, 0, &notifyVa);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker(halNotifyGetInfo, 10, 1);
    ret = RaHdcNotifyBaseAddrInit(NOTIFY, 0, &notifyVa);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker(halMemAlloc, 10, 1);
    ret = RaHdcNotifyBaseAddrInit(NOTIFY, 0, &notifyVa);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker(RaHdcNotifyCfgSet, 5, -1);
    mocker(halMemFree, 10, -1);
    ret = RaHdcNotifyBaseAddrInit(NOTIFY, 0, &notifyVa);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    struct RaRdmaHandle rdmaHandleTmp = {0};
    struct RaRdmaOps ops = {0};
    rdmaHandleTmp.rdevInfo.phyId = 0;
    rdmaHandleTmp.rdevInfo.family = AF_INET;
    rdmaHandleTmp.rdevInfo.family = 0;
    rdmaHandleTmp.rdmaOps = &ops;
    mocker(RaInetPton, 1, -1);
    ret = RaRdevDeinit(&rdmaHandleTmp, NOTIFY);
    EXPECT_INT_EQ(128000, ret);
    mocker_clean();

    mocker(RaInetPton, 1, 0);
    ret = RaRdevDeinit(&rdmaHandleTmp, NOTIFY);
    EXPECT_INT_EQ(128003, ret);
    mocker_clean();

    free(addr);
    free(data);
    free(wr);
    free(hdcSocketHandle);
    DlHalDeinit();
    return;
}

int RaRecvWrlistStub(
    struct RaQpHandle* handle, struct RecvWrlistData* wr, unsigned int recvNum, unsigned int* completeNum)
{
    return 0;
}
void TcRaRecvWrlist(void)
{
    int ret;
    struct RaQpHandle qpHandle = {0};
    struct RecvWrlistData wr = {0};
    unsigned int recvNum = 1;
    unsigned int completeNum = 0;

    ret = RaRecvWrlist(NULL, NULL, recvNum, &completeNum);
    EXPECT_INT_EQ(128103, ret);

    wr.memList.len = 0xffffffff;
    ret = RaRecvWrlist(&qpHandle, &wr, recvNum, &completeNum);
    EXPECT_INT_EQ(128103, ret);

    wr.memList.len = 100;
    qpHandle.rdmaOps = NULL;
    ret = RaRecvWrlist(&qpHandle, &wr, recvNum, &completeNum);
    EXPECT_INT_EQ(128103, ret);

    struct RaRdmaOps rdmaOps = {0};
    rdmaOps.raRecvWrlist = RaRecvWrlistStub;
    qpHandle.rdmaOps = &rdmaOps;
    ret = RaRecvWrlist(&qpHandle, &wr, recvNum, &completeNum);
    EXPECT_INT_EQ(0, ret);
    return;
}

void TcHostRaSendWrlistExt()
{
    struct RaQpHandle qpHandle = {0};
    qpHandle.rdmaOps = NULL;

    struct SendWrlistDataExt wr[1] = {{0}};
    struct SendWrRsp opRsp[1] = {{0}};

    unsigned int completeNum;

    RaSendWrlistExt(&qpHandle, wr, opRsp, 1, &completeNum);
}

int RaSendNormalWrlistStub(
    struct RaQpHandle* qpHandle, struct WrInfo wr[], struct SendWrRsp opRsp[], struct WrlistSendCompleteNum wrlistNum)
{
    return 0;
}

void TcHostRaSendNormalWrlist()
{
    struct RaRdmaOps rdmaOps = {0};
    struct RaQpHandle qpHandle = {0};
    struct SendWrRsp opRsp[1] = {{0}};
    qpHandle.rdmaOps = NULL;
    unsigned int completeNum;
    struct WrInfo wr[1] = {{0}};
    int ret = 0;

    ret = RaSendNormalWrlist(&qpHandle, wr, opRsp, 1, &completeNum);
    EXPECT_INT_EQ(128103, ret);

    rdmaOps.raSendNormalWrlist = RaSendNormalWrlistStub;
    qpHandle.rdmaOps = &rdmaOps;
    wr[0].memList.len = MAX_SG_LIST_LEN_MAX + 1;
    ret = RaSendNormalWrlist(&qpHandle, wr, opRsp, 1, &completeNum);
    EXPECT_INT_EQ(128103, ret);

    wr[0].memList.len = 1;
    ret = RaSendNormalWrlist(&qpHandle, wr, opRsp, 1, &completeNum);
    EXPECT_INT_EQ(0, ret);

    return;
}

int RaSetQpAttrQosStub(struct RaQpHandle* qpStub, struct QosAttr* attr) { return 0; }

void TcRaSetQpAttrQos()
{
    int ret;
    struct QosAttr attr = {0};
    struct RaQpHandle qpHandle = {0};
    qpHandle.rdmaOps = NULL;

    ret = RaSetQpAttrQos(NULL, &attr);
    EXPECT_INT_EQ(128103, ret);

    ret = RaSetQpAttrQos(&qpHandle, NULL);
    EXPECT_INT_EQ(128103, ret);

    attr.tc = 256;
    attr.sl = 8;
    ret = RaSetQpAttrQos(&qpHandle, &attr);
    EXPECT_INT_EQ(128103, ret);

    attr.sl = 4;
    ret = RaSetQpAttrQos(&qpHandle, &attr);
    EXPECT_INT_EQ(128103, ret);

    attr.tc = 33 * 4;
    ret = RaSetQpAttrQos(&qpHandle, &attr);
    EXPECT_INT_EQ(128103, ret);

    struct RaRdmaOps rdmaOps = {0};
    rdmaOps.raSetQpAttrQos = RaSetQpAttrQosStub;
    qpHandle.rdmaOps = &rdmaOps;
    ret = RaSetQpAttrQos(&qpHandle, &attr);
    EXPECT_INT_EQ(0, ret);

    return;
}

int RaSetQpAttrTimeoutStub(struct RaQpHandle* qpStub, unsigned int* attr) { return 0; }

void TcRaSetQpAttrTimeout()
{
    int ret;
    unsigned int timeout = 0;
    struct RaQpHandle qpHandle = {0};
    qpHandle.rdmaOps = NULL;

    ret = RaSetQpAttrTimeout(NULL, &timeout);
    EXPECT_INT_EQ(128103, ret);

    ret = RaSetQpAttrTimeout(&qpHandle, NULL);
    EXPECT_INT_EQ(128103, ret);

    timeout = 4;
    ret = RaSetQpAttrTimeout(&qpHandle, &timeout);
    EXPECT_INT_EQ(128103, ret);

    timeout = 4;
    ret = RaSetQpAttrTimeout(&qpHandle, &timeout);
    EXPECT_INT_EQ(128103, ret);

    timeout = 23;
    ret = RaSetQpAttrTimeout(&qpHandle, &timeout);
    EXPECT_INT_EQ(128103, ret);

    struct RaRdmaOps rdmaOps = {0};
    rdmaOps.raSetQpAttrTimeout = RaSetQpAttrTimeoutStub;
    qpHandle.rdmaOps = &rdmaOps;
    ret = RaSetQpAttrTimeout(&qpHandle, &timeout);
    EXPECT_INT_EQ(0, ret);

    return;
}

int RaSetQpAttrRetryCntStub(struct RaQpHandle* qpStub, unsigned int* retryCnt) { return 0; }

void TcRaSetQpAttrRetryCnt()
{
    int ret;
    unsigned int retryCnt = 0;
    struct RaQpHandle qpHandle = {0};
    qpHandle.rdmaOps = NULL;

    ret = RaSetQpAttrRetryCnt(NULL, &retryCnt);
    EXPECT_INT_EQ(128103, ret);

    ret = RaSetQpAttrRetryCnt(&qpHandle, NULL);
    EXPECT_INT_EQ(128103, ret);

    retryCnt = 8;
    ret = RaSetQpAttrRetryCnt(&qpHandle, &retryCnt);
    EXPECT_INT_EQ(128103, ret);

    retryCnt = 4;
    ret = RaSetQpAttrRetryCnt(&qpHandle, &retryCnt);
    EXPECT_INT_EQ(128103, ret);

    retryCnt = 7;
    ret = RaSetQpAttrRetryCnt(&qpHandle, &retryCnt);
    EXPECT_INT_EQ(128103, ret);

    struct RaRdmaOps rdmaOps = {0};
    rdmaOps.raSetQpAttrRetryCnt = RaSetQpAttrRetryCntStub;
    qpHandle.rdmaOps = &rdmaOps;
    ret = RaSetQpAttrRetryCnt(&qpHandle, &retryCnt);
    EXPECT_INT_EQ(0, ret);

    return;
}

void TcRaCreateEventHandle(void)
{
    int ret;
    int fd;

    ret = RaCreateEventHandle(NULL);
    EXPECT_INT_NE(0, ret);

    ret = RaCreateEventHandle(&fd);
    EXPECT_INT_EQ(0, ret);

    mocker(RaPeerCreateEventHandle, 1024, -EINVAL);
    ret = RaCreateEventHandle(&fd);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

void TcRaCtlEventHandle(void)
{
    int ret;
    int fd = 0;
    int fdHandle;

    ret = RaCtlEventHandle(-1, NULL, 4, 100);
    EXPECT_INT_NE(0, ret);

    ret = RaCtlEventHandle(fd, NULL, 4, 100);
    EXPECT_INT_NE(0, ret);

    ret = RaCtlEventHandle(fd, &fdHandle, 4, 100);
    EXPECT_INT_NE(0, ret);

    ret = RaCtlEventHandle(fd, &fdHandle, 1, 100);
    EXPECT_INT_NE(0, ret);

    ret = RaCtlEventHandle(fd, &fdHandle, 1, 0);
    EXPECT_INT_EQ(0, ret);

    mocker(RaPeerCtlEventHandle, 1024, -EINVAL);
    ret = RaCtlEventHandle(fd, &fdHandle, 1, 0);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

void TcRaWaitEventHandle(void)
{
    int ret;
    int fd = 0;
    unsigned int eventsNum = 0;
    struct SocketEventInfoT eventInfo = {};

    ret = RaWaitEventHandle(-1, NULL, -2, -1, NULL);
    EXPECT_INT_NE(0, ret);

    ret = RaWaitEventHandle(fd, NULL, -2, -1, NULL);
    EXPECT_INT_NE(0, ret);

    ret = RaWaitEventHandle(fd, &eventInfo, -2, -1, NULL);
    EXPECT_INT_NE(0, ret);

    ret = RaWaitEventHandle(fd, &eventInfo, 0, -1, NULL);
    EXPECT_INT_NE(0, ret);

    ret = RaWaitEventHandle(fd, &eventInfo, 0, 1, NULL);
    EXPECT_INT_NE(0, ret);

    ret = RaWaitEventHandle(fd, &eventInfo, 0, 1, &eventsNum);
    EXPECT_INT_EQ(0, ret);

    mocker(RaPeerWaitEventHandle, 1024, -EINVAL);
    ret = RaWaitEventHandle(fd, &eventInfo, 0, 1, &eventsNum);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

void TcRaDestroyEventHandle(void)
{
    int ret;
    int fd;

    ret = RaDestroyEventHandle(NULL);
    EXPECT_INT_NE(0, ret);

    ret = RaDestroyEventHandle(&fd);
    EXPECT_INT_EQ(0, ret);

    mocker(RaPeerDestroyEventHandle, 1024, -EINVAL);
    ret = RaDestroyEventHandle(&fd);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

int RaHdcPollCqStub(struct RaQpHandle* qpHdc, bool isSendCq, unsigned int numEntries, void* wc)
{
    if (isSendCq) {
        return -1;
    }
    return 0;
}

void TcRaPollCq(void)
{
    int ret;
    struct RaQpHandle qpHandle = {0};
    struct RaRdmaOps rdmaOps = {0};
    struct rdma_lite_wc_v2 liteWc = {0};

    ret = RaPollCq(NULL, true, 0, NULL);
    EXPECT_INT_NE(0, ret);

    qpHandle.rdmaOps = &rdmaOps;
    rdmaOps.raPollCq = NULL;
    ret = RaPollCq(&qpHandle, true, 1, &liteWc);
    EXPECT_INT_NE(0, ret);

    rdmaOps.raPollCq = RaHdcPollCqStub;
    ret = RaPollCq(&qpHandle, true, 1, &liteWc);
    EXPECT_INT_NE(0, ret);

    ret = RaPollCq(&qpHandle, false, 1, &liteWc);
    EXPECT_INT_EQ(0, ret);
}

void TcGetVnicIpInfos(void)
{
    int ret;
    unsigned int ids[1] = {0};
    struct IpInfo infos[1] = {0};

    ret = RaSocketGetVnicIpInfos(0, PHY_ID_VNIC_IP, NULL, 0, NULL);
    EXPECT_INT_NE(0, ret);

    ret = RaSocketGetVnicIpInfos(0xFFFF, PHY_ID_VNIC_IP, ids, 1, infos);
    EXPECT_INT_NE(0, ret);

    ret = RaSocketGetVnicIpInfos(0, PHY_ID_VNIC_IP, ids, 1, infos);
    EXPECT_INT_NE(0, ret);

    ret = RaSocketGetVnicIpInfos(0, 0xFFFF, ids, 1, infos);
    EXPECT_INT_NE(0, ret);

    gInterfaceVersion = 0;
    mocker_invoke(RaHdcGetInterfaceVersion, RaGetInterfaceVersionStub, 100);
    ret = RaSocketGetVnicIpInfos(0, PHY_ID_VNIC_IP, ids, 1, infos);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

int RaSocketBatchAbortStub(unsigned int phyId, struct SocketConnectInfoT conn[], unsigned int num) { return 0; }

void TcRaSocketBatchAbort(void)
{
    int ret;
    unsigned int phyId = 0;
    struct SocketConnectInfoT conn = {0};
    struct RaSocketHandle socketHandle = {0};
    struct RaSocketOps socketOps = {0};

    ret = RaSocketBatchAbort(NULL, 0);
    EXPECT_INT_EQ(128203, ret);

    conn.socketHandle = NULL;
    ret = RaSocketBatchAbort(&conn, 1);
    EXPECT_INT_EQ(128203, ret);

    socketOps.raSocketBatchAbort = RaSocketBatchAbortStub;
    socketHandle.socketOps = &socketOps;
    socketHandle.rdevInfo.phyId = 16;
    conn.socketHandle = &socketHandle;
    ret = RaSocketBatchAbort(&conn, 1);
    EXPECT_INT_EQ(128203, ret);

    socketHandle.rdevInfo.phyId = 0;
    mocker(RaInetPton, 5, -1);
    ret = RaSocketBatchAbort(&conn, 1);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    mocker(RaInetPton, 5, 0);
    ret = RaSocketBatchAbort(&conn, 1);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRaGetClientSocketErrInfo(void)
{
    int ret = 0;
    struct SocketConnectInfoT conn[10] = {0};
    struct SocketErrInfo err[10] = {0};
    unsigned int num = 1;
    struct RaSocketHandle* socketHandle = NULL;
    struct rdev rdevInfo = {0};

    socketHandle = malloc(sizeof(struct RaSocketHandle));
    socketHandle->rdevInfo = rdevInfo;
    mocker_clean();

    conn[0].socketHandle = NULL;
    ret = RaGetClientSocketErrInfo(conn, err, num);
    EXPECT_INT_EQ(128203, ret);

    conn[0].socketHandle = socketHandle;
    socketHandle->socketOps = &gRaPeerSocketOps;
    rdevInfo.phyId = 0;
    mocker(RaInetPton, 5, 0);
    mocker(RaGetSocketConnectInfo, 1, 0);
    mocker(RsSetCtx, 1, 0);
    mocker(RsSocketGetClientSocketErrInfo, 1, 1);
    ret = RaGetClientSocketErrInfo(conn, err, num);
    EXPECT_INT_EQ(328207, ret);
    mocker_clean();

    mocker(RaInetPton, 5, 0);
    mocker(RaGetSocketConnectInfo, 1, 0);
    mocker(RsSetCtx, 1, 0);
    mocker(RsSocketGetClientSocketErrInfo, 1, 0);
    ret = RaGetClientSocketErrInfo(conn, err, num);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    free(socketHandle);
    socketHandle = NULL;
}

void TcRaGetServerSocketErrInfo(void)
{
    int ret = 0;
    struct SocketListenInfoT conn[10] = {0};
    struct ServerSocketErrInfo err[10] = {0};
    unsigned int num = 1;
    struct RaSocketHandle* socketHandle = NULL;
    struct rdev rdevInfo = {0};

    socketHandle = malloc(sizeof(struct RaSocketHandle));
    socketHandle->rdevInfo = rdevInfo;
    mocker_clean();

    conn[0].socketHandle = NULL;
    ret = RaGetServerSocketErrInfo(conn, err, num);
    EXPECT_INT_EQ(128203, ret);

    conn[0].socketHandle = socketHandle;
    socketHandle->socketOps = &gRaPeerSocketOps;
    rdevInfo.phyId = 0;
    mocker(RaInetPton, 5, 0);
    mocker(RaGetSocketListenInfo, 1, 0);
    mocker(RsSetCtx, 1, 0);
    mocker(RsSocketGetServerSocketErrInfo, 1, 1);
    ret = RaGetServerSocketErrInfo(conn, err, num);
    EXPECT_INT_EQ(328207, ret);
    mocker_clean();

    mocker(RaInetPton, 5, 0);
    mocker(RaGetSocketListenInfo, 1, 0);
    mocker(RsSetCtx, 1, 0);
    mocker(RsSocketGetServerSocketErrInfo, 1, 0);
    ret = RaGetServerSocketErrInfo(conn, err, num);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    free(socketHandle);
    socketHandle = NULL;
}

void TcRaSocketAcceptCreditAdd(void)
{
    struct SocketListenInfoT conn[10] = {0};
    struct RaSocketHandle socketHandle = {0};
    struct RaSocketOps socketOps = {0};
    int ret = 0;

    conn[0].socketHandle = &socketHandle;
    socketHandle.socketOps = &socketOps;
    socketHandle.socketOps->raSocketAcceptCreditAdd = RaPeerSocketAcceptCreditAdd;
    mocker(RaInetPton, 1, 0);
    mocker(RaGetSocketListenInfo, 1, 0);
    mocker(RsSetCtx, 1, 0);
    mocker(RaPeerMutexLock, 1, 0);
    mocker(RaPeerMutexUnlock, 1, 0);
    mocker(RsSocketAcceptCreditAdd, 1, 0);
    ret = RaSocketAcceptCreditAdd(conn, 1, 1);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRaRemapMr(void)
{
    struct RaRdmaHandle rdmaHandle = {0};
    struct MemRemapInfo info[1] = {0};
    struct RaRdmaOps rdmaOps = {0};
    int ret = 0;

    mocker_clean();
    rdmaHandle.rdmaOps = &rdmaOps;
    rdmaHandle.rdmaOps->raRemapMr = RaHdcRemapMr;
    mocker(RaHdcProcessMsg, 1, 0);
    ret = RaRemapMr((void*)&rdmaHandle, info, 1);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRaRegisterMr(void)
{
    struct RaRdmaHandle rdmaHandle = {0};
    struct RaMrHandle* mrHandle = NULL;
    struct RaRdmaOps rdmaOps = {0};
    struct MrInfoT mrInfo = {0};
    int ret = 0;

    mocker_clean();
    mocker(RaHdcProcessMsg, 2, 0);
    rdmaHandle.rdmaOps = &rdmaOps;
    rdmaHandle.rdmaOps->raRegisterMr = RaHdcTypicalMrReg;
    rdmaHandle.rdmaOps->raDeregisterMr = RaHdcTypicalMrDereg;
    ret = RaRegisterMr((void*)&rdmaHandle, &mrInfo, (void**)&mrHandle);
    EXPECT_INT_EQ(0, ret);
    ret = RaDeregisterMr((void*)&rdmaHandle, (void*)mrHandle);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRaGetLbMax(void)
{
    struct RaRdmaHandle rdmaHandle = {0};
    struct RaRdmaOps rdmaOps = {0};
    int lbMax = 0;
    int ret = 0;

    ret = RaGetLbMax(NULL, &lbMax);
    EXPECT_INT_EQ(128103, ret);

    ret = RaGetLbMax(&rdmaHandle, NULL);
    EXPECT_INT_EQ(128103, ret);

    ret = RaGetLbMax(&rdmaHandle, &lbMax);
    EXPECT_INT_EQ(128103, ret);

    rdmaHandle.rdmaOps = &rdmaOps;
    ret = RaGetLbMax(&rdmaHandle, &lbMax);
    EXPECT_INT_EQ(128103, ret);

    rdmaHandle.rdmaOps->raGetLbMax = RaPeerGetLbMax;
    ret = RaGetLbMax(&rdmaHandle, &lbMax);
    EXPECT_INT_EQ(0, ret);
}

void TcRaSetQpLbValue(void)
{
    struct RaQpHandle qpHandle = {0};
    struct RaRdmaOps rdmaOps = {0};
    int lbvalue = 0;
    int ret = 0;

    ret = RaSetQpLbValue(NULL, lbvalue);
    EXPECT_INT_EQ(128103, ret);

    ret = RaSetQpLbValue(&qpHandle, lbvalue);
    EXPECT_INT_EQ(128103, ret);

    qpHandle.rdmaOps = &rdmaOps;
    ret = RaSetQpLbValue(&qpHandle, lbvalue);
    EXPECT_INT_EQ(128103, ret);

    qpHandle.rdmaOps->raSetQpLbValue = RaPeerSetQpLbValue;
    ret = RaSetQpLbValue(&qpHandle, lbvalue);
    EXPECT_INT_EQ(0, ret);
}

void TcRaGetQpLbValue(void)
{
    struct RaQpHandle qpHandle = {0};
    struct RaRdmaOps rdmaOps = {0};
    int lbvalue = 0;
    int ret = 0;

    ret = RaGetQpLbValue(NULL, &lbvalue);
    EXPECT_INT_EQ(128103, ret);

    ret = RaGetQpLbValue(&qpHandle, NULL);
    EXPECT_INT_EQ(128103, ret);

    ret = RaGetQpLbValue(&qpHandle, &lbvalue);
    EXPECT_INT_EQ(128103, ret);

    qpHandle.rdmaOps = &rdmaOps;
    ret = RaGetQpLbValue(&qpHandle, &lbvalue);
    EXPECT_INT_EQ(128103, ret);

    qpHandle.rdmaOps->raGetQpLbValue = RaPeerGetQpLbValue;
    ret = RaGetQpLbValue(&qpHandle, &lbvalue);
    EXPECT_INT_EQ(0, ret);
}

int RaGetQpHyperFeatureFailStub(struct RaQpHandle* qpStub, struct HyperFeature* hyperFeature) { return -1; }

void TcRaGetQpHyperFeature(void)
{
    struct RaRdmaHandle rdmaHandle = {0};
    struct RaQpHandle qpHandle = {0};
    struct RaRdmaOps rdmaOps = {0};
    struct QpAttr attr = {0};
    int ret = 0;

    qpHandle.rdmaHandle = &rdmaHandle;

    ret = RaGetQpAttr(NULL, &attr);
    EXPECT_INT_EQ(128103, ret);

    ret = RaGetQpAttr(&qpHandle, NULL);
    EXPECT_INT_EQ(128103, ret);

    ret = RaGetQpAttr(&qpHandle, &attr);
    EXPECT_INT_EQ(0, ret);

    qpHandle.rdmaOps = &rdmaOps;
    ret = RaGetQpAttr(&qpHandle, &attr);
    EXPECT_INT_EQ(0, ret);

    qpHandle.rdmaOps->raGetQpHyperFeature = RaGetQpHyperFeatureStub;
    ret = RaGetQpAttr(&qpHandle, &attr);
    EXPECT_INT_EQ(0, ret);

    qpHandle.rdmaOps->raGetQpHyperFeature = RaGetQpHyperFeatureFailStub;
    ret = RaGetQpAttr(&qpHandle, &attr);
    EXPECT_INT_EQ(0, ret);
}

void TcRaSetDebugConfig()
{
    int ret;
    struct RaInfo info = {0};

    /* 参数校验：info 为空 */
    ret = RaSetDebugConfig(NULL, 0);
    EXPECT_INT_NE(0, ret);

    /* 参数校验：phyId 越界 */
    info.mode = NETWORK_PEER_ONLINE;
    info.phyId = RA_MAX_PHY_ID_NUM;
    ret = RaSetDebugConfig(&info, 0);
    EXPECT_INT_NE(0, ret);

    /* PEER_ONLINE：直写本进程调试配置，校验配置生效 */
    info.phyId = 0;
    info.mode = NETWORK_PEER_ONLINE;
    ret = RaSetDebugConfig(&info, 0xF);
    EXPECT_INT_EQ(0, ret);
    EXPECT_INT_EQ(0xF, (int)HccpGetDebugConfig());

    /* OFFLINE：经 RaHdcSetDebugConfig 下发（mock 成功） */
    info.mode = NETWORK_OFFLINE;
    mocker(RaHdcSetDebugConfig, 1, 0);
    ret = RaSetDebugConfig(&info, 0xF);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    /* 非法 mode：返回 EPROTONOSUPPORT */
    info.mode = 99;
    ret = RaSetDebugConfig(&info, 0xF);
    EXPECT_INT_NE(0, ret);

    return;
}
