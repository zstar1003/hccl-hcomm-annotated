/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <errno.h>
#include "ascend_hal.h"
#include "dl_hal_function.h"
#include "dl_urma_function.h"
#include "rs_socket.h"
#include "ut_dispatch.h"
#include "ra_rs_comm.h"
#include "rs.h"
#include "hccp_common.h"
#include "rs_inner.h"
#include "rs_ping_inner.h"
#include "hccp_ping.h"
#include "rs_epoll.h"
#include "rs_ping.h"
#include "rs_ping_urma.h"
#include "tc_ut_rs_ping_urma.h"

extern int RsGetPingCb(struct RaRsDevInfo* rdev, struct RsPingCtxCb** pingCb);
extern int RsPingUrmaPollRcq(struct RsPingCtxCb* pingCb, int* polledCnt, struct timeval* timestamp2);
extern void RsPongUrmaHandleSend(struct RsPingCtxCb* pingCb, int polledCnt, struct timeval* timestamp2);
extern void RsPongUrmaPollRcq(struct RsPingCtxCb* pingCb);
extern int RsPingUrmaPollScq(struct RsPingCtxCb* pingCb, struct RsPingTargetInfo* target);
extern struct RsPingPongOps* RsPingUrmaGetOps(void);
extern struct RsPingPongDfx* RsPingUrmaGetDfx(void);
extern int
RsPingCommonImportJetty(urma_context_t* urmaCtx, struct PingQpInfo* target, urma_target_jetty_t** importTjetty);
extern int RsPongJettyPostSend(struct RsPingCtxCb* pingCb, urma_cr_t* cr, struct timeval* timestamp2);
extern int RsPingCommonJfrPostRecv(struct RsPingLocalJettyCb* jettyCb);
extern int RsPongJettyResolveResponsePacket(struct RsPingCtxCb* pingCb, uint32_t sgeIdx, struct timeval* timestamp4);
extern int
RsPingUrmaFindTargetNode(struct RsPingCtxCb* pingCb, struct PingQpInfo* target, struct RsPingTargetInfo** node);
extern int RsPingCommonPollSendJfc(struct RsPingLocalJettyCb* jettyCb);
extern int
RsPongJettyFindAllocTargetNode(struct RsPingCtxCb* pingCb, struct PingQpInfo* target, struct RsPongTargetInfo** node);
extern int
RsPongJettyFindTargetNode(struct RsPingCtxCb* pingCb, struct PingQpInfo* target, struct RsPongTargetInfo** node);
extern int RsGetJettyInfo(struct PingQpInfo* qpInfo, urma_jetty_id_t* jettyId, urma_eid_t* eid);
extern int RsPingUrmaPostSend(struct RsPingCtxCb* pingCb, struct RsPingTargetInfo* target);
extern void RsPingUrmaResetRecvBuffer(struct RsPingCtxCb* pingCb);

urma_jfc_t gTmpJfc;
static struct rs_cb gTmpRsCb;
static struct RsPingCtxCb gTmpPingCb;
static struct RsPingTargetInfo gTmpTarget;
static struct RsPingTargetInfo gTmpTarget1;

#define TEST_SGE_LIST_LEN 1024

int RsGetRsCbUrmaStub(unsigned int phyId, struct rs_cb** rsCb)
{
    *rsCb = &gTmpRsCb;
    (*rsCb)->pingCb.pingPongOps = RsPingUrmaGetOps();
    (*rsCb)->pingCb.pingPongDfx = RsPingUrmaGetDfx();
    return 0;
}

int RsGetPingCbUrmaStub(struct RaRsDevInfo* rdev, struct RsPingCtxCb** pingCb)
{
    *pingCb = &gTmpPingCb;
    (*pingCb)->pingPongOps = RsPingUrmaGetOps();
    (*pingCb)->pingPongDfx = RsPingUrmaGetDfx();
    return 0;
}

int RsGetPingCbUrmaStub1(struct RaRsDevInfo* rdev, struct RsPingCtxCb** pingCb)
{
    *pingCb = &gTmpRsCb.pingCb;
    (*pingCb)->pingPongOps = RsPingUrmaGetOps();
    (*pingCb)->pingPongDfx = RsPingUrmaGetDfx();
    return 0;
}

int RsUrmaPollJfcStubPing0(urma_jfc_t* jfc, int crCnt, urma_cr_t* cr)
{
    cr->status = URMA_CR_LOC_LEN_ERR;
    return 1;
}

int RsUrmaPollJfcStubPing1(urma_jfc_t* jfc, int crCnt, urma_cr_t* cr)
{
    cr->status = URMA_CR_SUCCESS;
    return 1;
}

int RsUrmaWaitJfcStub(urma_jfce_t* jfce, uint32_t jfcCnt, int timeOut, urma_jfc_t* jfc[])
{
    *jfc = &gTmpJfc;
    return 1;
}

int RsPingUrmaFindTargetNodeStub(struct RsPingCtxCb* pingCb, struct PingQpInfo* target, struct RsPingTargetInfo** node)
{
    *node = &gTmpTarget;

    return -ENODEV;
}

int RsPingUrmaFindTargetNodeStub1(struct RsPingCtxCb* pingCb, struct PingQpInfo* target, struct RsPingTargetInfo** node)
{
    *node = &gTmpTarget1;

    return 0;
}

int RsPongJettyFindAllocTargetNodeStub(
    struct RsPingCtxCb* pingCb, struct PingQpInfo* target, struct RsPongTargetInfo** node)
{
    static struct RsPongTargetInfo tmpNode = {0};

    *node = &tmpNode;

    return 0;
}

int RsPongJettyFindTargetNodeStub(struct RsPingCtxCb* pingCb, struct PingQpInfo* target, struct RsPongTargetInfo** node)
{
    *node = NULL;
    return -19;
}

int RsPongJettyFindTargetNodeStub2(
    struct RsPingCtxCb* pingCb, struct PingQpInfo* target, struct RsPongTargetInfo** node)
{
    *node = calloc(1, sizeof(struct RsPongTargetInfo));
    RS_INIT_LIST_HEAD(&(*node)->list);
    return 0;
}

void TcRsPingInitDeinitUrma()
{
    struct ibv_comp_channel clientChannel = {0};
    struct ibv_comp_channel serverChannel = {0};
    struct PingTargetInfo target = {0};
    struct PingInitAttr attr = {0};
    struct PingInitInfo info = {0};
    struct RaRsDevInfo rdev = {0};
    unsigned int rdevIndex = 0;
    int ret = 0;

    mocker_invoke(RsGetRsCb, RsGetRsCbUrmaStub, 20);
    mocker(rsGetLocalDevIDByHostDevID, 20, 0);
    mocker(RsSetupSharemem, 20, 0);
    mocker(RsEpollCtl, 20, 0);
    mocker(DlHalBuffAllocAlignEx, 20, 0);
    attr.protocol = PROTOCOL_UDMA;
    ret = RsPingInit(&attr, &info, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(info.result.headerSize, RS_PING_PAYLOAD_HEADER_RESV_CUSTOM);
    mocker_clean();

    gTmpRsCb.pingCb.initCnt = 1;
    RS_INIT_LIST_HEAD(&gTmpRsCb.pingCb.pingList);
    RS_INIT_LIST_HEAD(&gTmpRsCb.pingCb.pongList);
    target.payload.size = 1;
    mocker_invoke(RsGetPingCb, RsGetPingCbUrmaStub1, 10);
    mocker_invoke(RsGetRsCb, RsGetRsCbUrmaStub, 20);
    ret = RsPingTargetAdd(&rdev, &target);
    EXPECT_INT_EQ(ret, 0);
    mocker(RsEpollCtl, 20, 0);
    ret = RsPingDeinit(&rdev);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsPingTargetAddDelUrma()
{
    struct PingTargetInfo target = {0};
    struct RaRsDevInfo rdev = {0};
    unsigned int num = 1;
    int ret = 0;

    gTmpPingCb.taskStatus = RS_PING_TASK_RESET;
    RS_INIT_LIST_HEAD(&gTmpPingCb.pingList);
    mocker_invoke(RsGetPingCb, RsGetPingCbUrmaStub, 2);
    ret = RsPingTargetAdd(&rdev, &target);
    EXPECT_INT_EQ(ret, 0);
    ret = RsPingTargetDel(&rdev, (struct PingTargetCommInfo*)&target, &num);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    target.payload.size = 1;
    gTmpPingCb.taskStatus = RS_PING_TASK_RESET;
    mocker_invoke(RsGetPingCb, RsGetPingCbUrmaStub, 1);
    mocker(RsPingCommonImportJetty, 1, -1);
    ret = RsPingTargetAdd(&rdev, &target);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker_invoke(RsGetPingCb, RsGetPingCbUrmaStub, 1);
    mocker(calloc, 10, 0);
    ret = RsPingTargetAdd(&rdev, &target);
    EXPECT_INT_EQ(ret, -12);
    mocker_clean();

    gTmpPingCb.taskStatus = RS_PING_TASK_RESET;
    mocker_invoke(RsGetPingCb, RsGetPingCbUrmaStub, 1);
    mocker(RsPingUrmaFindTargetNode, 1, -1);
    ret = RsPingTargetDel(&rdev, (struct PingTargetCommInfo*)&target, &num);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRsPingUrmaPostSend()
{
    struct RsPingTargetInfo target = {0};
    struct RsPingCtxCb pingCb = {0};
    urma_jetty_t serverJetty = {0};
    urma_jetty_t clientJetty = {0};
    void* addr = malloc(256);
    urma_sge_t sge = {0};
    int ret = 0;

    target.payloadBuffer = malloc(1);
    target.payloadSize = 1;
    sge.addr = (uintptr_t)addr;
    sge.len = 256;
    pingCb.pingJetty.sendSegCb.sgeNum = 1;
    pingCb.pingJetty.sendSegCb.sgeList = &sge;
    pingCb.pongJetty.jetty = &serverJetty;
    pingCb.pingJetty.jetty = &clientJetty;
    ret = RsPingUrmaPostSend(&pingCb, &target);
    EXPECT_INT_EQ(ret, 0);
    free(addr);
    addr = NULL;
    free(target.payloadBuffer);
    target.payloadBuffer = NULL;
}

void TcRsPingClientPollCqUrma()
{
    struct RsPingCtxCb pingCb = {0};
    struct timeval timestamp2 = {0};
    int polledCnt = 0;
    int ret = 0;

    mocker_clean();
    ret = RsPingUrmaPollRcq(&pingCb, &polledCnt, &timestamp2);
    EXPECT_INT_EQ(ret, -11);
    mocker(RsUrmaWaitJfc, 1, -1);
    ret = RsPingUrmaPollRcq(&pingCb, &polledCnt, &timestamp2);
    EXPECT_INT_EQ(ret, -259);
    mocker_clean();

    mocker(RsUrmaWaitJfc, 1, 1);
    mocker(RsUrmaAckJfc, 1, 0);
    ret = RsPingUrmaPollRcq(&pingCb, &polledCnt, &timestamp2);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    pingCb.pingJetty.recvJfc.jfc = &gTmpJfc;
    mocker_invoke(RsUrmaWaitJfc, RsUrmaWaitJfcStub, 1);
    mocker(RsUrmaPollJfc, 1, -1);
    ret = RsPingUrmaPollRcq(&pingCb, &polledCnt, &timestamp2);
    EXPECT_INT_EQ(ret, -259);
    mocker_clean();

    pingCb.pingJetty.recvJfc.maxRecvWcNum = 16;
    mocker_invoke(RsUrmaWaitJfc, RsUrmaWaitJfcStub, 1);
    mocker(RsUrmaPollJfc, 1, 1);
    mocker(RsPongJettyPostSend, 1, -1);
    ret = RsPingUrmaPollRcq(&pingCb, &polledCnt, &timestamp2);
    EXPECT_INT_EQ(ret, 0);
    RsPongUrmaHandleSend(&pingCb, polledCnt, &timestamp2);
    mocker_clean();

    mocker_invoke(RsUrmaWaitJfc, RsUrmaWaitJfcStub, 1);
    mocker(RsUrmaPollJfc, 1, 1);
    mocker(RsPongJettyPostSend, 1, 0);
    mocker(RsPingCommonJfrPostRecv, 1, -1);
    ret = RsPingUrmaPollRcq(&pingCb, &polledCnt, &timestamp2);
    EXPECT_INT_EQ(ret, 0);
    RsPongUrmaHandleSend(&pingCb, polledCnt, &timestamp2);
    mocker_clean();

    mocker_invoke(RsUrmaWaitJfc, RsUrmaWaitJfcStub, 1);
    mocker(RsUrmaPollJfc, 1, 1);
    mocker(RsPongJettyPostSend, 1, 0);
    mocker(RsPingCommonJfrPostRecv, 1, -1);
    mocker(RsUrmaRearmJfc, 1, -1);
    ret = RsPingUrmaPollRcq(&pingCb, &polledCnt, &timestamp2);
    EXPECT_INT_EQ(ret, 0);
    RsPongUrmaHandleSend(&pingCb, polledCnt, &timestamp2);
    mocker_clean();
}

void TcRsPingUrmaPollScq()
{
    struct RsPingTargetInfo target = {0};
    struct RsPingCtxCb pingCb = {0};
    int ret = 0;

    ret = RsPingUrmaPollScq(&pingCb, &target);
    EXPECT_INT_EQ(ret, -61);

    mocker_invoke(RsUrmaPollJfc, RsUrmaPollJfcStubPing0, 10);
    ret = RsPingUrmaPollScq(&pingCb, &target);
    EXPECT_INT_EQ(ret, -259);
    mocker_clean();

    mocker_invoke(RsUrmaPollJfc, RsUrmaPollJfcStubPing1, 10);
    ret = RsPingUrmaPollScq(&pingCb, &target);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsPingServerPollCqUrma()
{
    struct RsPingCtxCb pingCb = {0};

    mocker(RsUrmaWaitJfc, 1, -1);
    RsPongUrmaPollRcq(&pingCb);
    mocker_clean();

    mocker(RsUrmaWaitJfc, 1, 0);
    RsPongUrmaPollRcq(&pingCb);
    mocker_clean();

    pingCb.pongJetty.recvJfc.jfc = &gTmpJfc;
    pingCb.pongJetty.recvJfc.maxRecvWcNum = 16;
    mocker_invoke(RsUrmaWaitJfc, RsUrmaWaitJfcStub, 1);
    mocker(RsUrmaPollJfc, 1, -1);
    RsPongUrmaPollRcq(&pingCb);
    mocker_clean();

    mocker_invoke(RsUrmaWaitJfc, RsUrmaWaitJfcStub, 1);
    mocker(RsUrmaPollJfc, 1, 1);
    mocker(RsPongJettyResolveResponsePacket, 1, -1);
    RsPongUrmaPollRcq(&pingCb);
    mocker_clean();

    mocker_invoke(RsUrmaWaitJfc, RsUrmaWaitJfcStub, 1);
    mocker(RsUrmaPollJfc, 1, 1);
    mocker(RsPongJettyResolveResponsePacket, 1, 0);
    mocker(RsPingCommonJfrPostRecv, 1, -1);
    RsPongUrmaPollRcq(&pingCb);
    mocker_clean();

    mocker_invoke(RsUrmaWaitJfc, RsUrmaWaitJfcStub, 1);
    mocker(RsUrmaPollJfc, 1, 1);
    mocker(RsPongJettyResolveResponsePacket, 1, 0);
    mocker(RsPingCommonJfrPostRecv, 1, 0);
    mocker(RsUrmaRearmJfc, 1, -1);
    RsPongUrmaPollRcq(&pingCb);
    mocker_clean();
}

void TcRsEpollEventPingHandleUrma()
{
    urma_jfce_t pingJfce = {0};
    urma_jfce_t pongJfce = {0};
    struct rs_cb rscb = {0};
    int ret = 0;

    pongJfce.fd = 1;
    rscb.pingCb.initCnt = 1;
    rscb.pingCb.pingJetty.jfce = &pingJfce;
    rscb.pingCb.pongJetty.jfce = &pongJfce;
    rscb.pingCb.threadStatus = RS_PING_THREAD_RUNNING;
    rscb.pingCb.pingPongOps = RsPingUrmaGetOps();
    rscb.pingCb.pingPongDfx = RsPingUrmaGetDfx();

    mocker(RsPingUrmaPollRcq, 10, 0);
    mocker(RsPongUrmaHandleSend, 10, 0);
    mocker(RsPongUrmaPollRcq, 10, 0);

    ret = RsEpollEventPingHandle(&rscb, 0);
    EXPECT_INT_EQ(ret, 0);
    ret = RsEpollEventPingHandle(&rscb, 1);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
    return;
}

void TcRsPingGetResultsUrma()
{
    struct PingTargetCommInfo target = {0};
    struct PingResultInfo result = {0};
    struct RaRsDevInfo rdev = {0};
    unsigned int num = 1;
    int ret = 0;

    gTmpPingCb.taskStatus = RS_PING_TASK_RESET;
    mocker_invoke(RsGetPingCb, RsGetPingCbUrmaStub, 1);
    mocker_invoke(RsPingUrmaFindTargetNode, RsPingUrmaFindTargetNodeStub, 1);
    ret = RsPingGetResults(&rdev, &target, &num, &result);
    EXPECT_INT_EQ(ret, -ENODEV);
    mocker_clean();

    num = 1;
    gTmpPingCb.taskStatus = RS_PING_TASK_RESET;
    mocker_invoke(RsGetPingCb, RsGetPingCbUrmaStub, 1);
    mocker_invoke(RsPingUrmaFindTargetNode, RsPingUrmaFindTargetNodeStub1, 1);
    ret = RsPingGetResults(&rdev, &target, &num, &result);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsPingServerPostSendUrma()
{
    struct RsPingCtxCb pingCb = {0};
    struct timeval timestamp2 = {0};
    void* sendAddr = malloc(TEST_SGE_LIST_LEN);
    void* recvAddr = malloc(TEST_SGE_LIST_LEN);
    urma_cr_t cr = {0};
    int ret = 0;

    cr.user_ctx = 1;
    mocker(RsPingCommonPollSendJfc, 1, 0);
    ret = RsPongJettyPostSend(&pingCb, &cr, &timestamp2);
    EXPECT_INT_EQ(ret, -EIO);
    mocker_clean();

    cr.user_ctx = 0;
    cr.completion_len = 16;
    pingCb.pingJetty.recvSegCb.sgeList = calloc(1, sizeof(urma_sge_t));
    pingCb.pingJetty.recvSegCb.sgeNum = 1;
    pingCb.pongJetty.sendSegCb.sgeList = calloc(1, sizeof(urma_sge_t));
    pingCb.pongJetty.sendSegCb.sgeNum = 1;
    pingCb.pongJetty.sendSegCb.sgeList->addr = (uintptr_t)sendAddr;
    pingCb.pongJetty.sendSegCb.sgeList->len = TEST_SGE_LIST_LEN;
    pingCb.pingJetty.recvSegCb.sgeList->addr = (uintptr_t)recvAddr;
    pingCb.pingJetty.recvSegCb.sgeList->len = TEST_SGE_LIST_LEN;
    mocker(RsPingCommonPollSendJfc, 1, 0);
    mocker(RsPongJettyFindAllocTargetNode, 1, -1);
    ret = RsPongJettyPostSend(&pingCb, &cr, &timestamp2);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(RsPingCommonPollSendJfc, 1, 0);
    mocker_invoke(RsPongJettyFindAllocTargetNode, RsPongJettyFindAllocTargetNodeStub, 1);
    mocker(gettimeofday, 20, 1);
    mocker(RsUrmaPostJettySendWr, 1, -1);
    ret = RsPongJettyPostSend(&pingCb, &cr, &timestamp2);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(RsPingCommonPollSendJfc, 1, 0);
    mocker_invoke(RsPongJettyFindAllocTargetNode, RsPongJettyFindAllocTargetNodeStub, 1);
    ret = RsPongJettyPostSend(&pingCb, &cr, &timestamp2);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    free(pingCb.pongJetty.sendSegCb.sgeList);
    pingCb.pongJetty.sendSegCb.sgeList = NULL;
    free(pingCb.pingJetty.recvSegCb.sgeList);
    pingCb.pingJetty.recvSegCb.sgeList = NULL;
    free(recvAddr);
    recvAddr = NULL;
    free(sendAddr);
    sendAddr = NULL;
}

void TcRsPongJettyFindAllocTargetNode()
{
    struct RsPongTargetInfo* node = NULL;
    struct RsPingCtxCb pingCb = {0};
    struct PingQpInfo target = {0};
    int ret = 0;

    mocker_invoke(RsPongJettyFindTargetNode, RsPongJettyFindTargetNodeStub2, 1);
    mocker(RsPingCommonImportJetty, 1, -1);
    ret = RsPongJettyFindAllocTargetNode(&pingCb, &target, &node);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    pthread_mutex_init(&pingCb.pongMutex, NULL);
    RS_INIT_LIST_HEAD(&pingCb.pongList);
    mocker_invoke(RsPongJettyFindTargetNode, RsPongJettyFindTargetNodeStub, 1);
    mocker(RsPingCommonImportJetty, 1, 0);
    ret = RsPongJettyFindAllocTargetNode(&pingCb, &target, &node);
    free(node);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsPingCommonPollSendJfc()
{
    struct RsPingLocalQpCb qpCb = {0};
    int ret = 0;

    mocker(RsUrmaPollJfc, 1, -1);
    ret = RsPingCommonPollSendJfc((struct RsPingLocalJettyCb*)&qpCb);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker(RsUrmaPollJfc, 1, 1);
    ret = RsPingCommonPollSendJfc((struct RsPingLocalJettyCb*)&qpCb);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsPongJettyFindTargetNode()
{
    struct RsPongTargetInfo stubNode = {0};
    struct RsPongTargetInfo* node = NULL;
    struct RsPingCtxCb pingCb = {0};
    struct PingQpInfo target = {0};
    int ret = 0;

    RS_INIT_LIST_HEAD(&pingCb.pongList);
    ret = RsPongJettyFindTargetNode(&pingCb, &target, &node);
    EXPECT_INT_EQ(ret, -ENODEV);

    RsListAddTail(&stubNode.list, &pingCb.pongList);
    ret = RsPongJettyFindTargetNode(&pingCb, &target, &node);
    EXPECT_INT_EQ(ret, 0);
}

void TcRsPongJettyResolveResponsePacket()
{
    struct RsPingCtxCb pingCb = {0};
    struct timeval timestamp4 = {0};
    void* recvAddr = calloc(1, TEST_SGE_LIST_LEN);
    uint32_t sgeIdx = 0;
    int ret = 0;

    pingCb.pongJetty.recvSegCb.sgeList = calloc(1, sizeof(urma_sge_t));
    pingCb.pongJetty.recvSegCb.sgeList->addr = (uintptr_t)recvAddr;
    pingCb.pongJetty.recvSegCb.sgeList->len = TEST_SGE_LIST_LEN;
    pingCb.taskId = 1;

    ret = RsPongJettyResolveResponsePacket(&pingCb, sgeIdx, &timestamp4);
    EXPECT_INT_EQ(ret, 0);

    pingCb.taskId = 0;
    mocker_invoke(RsPingUrmaFindTargetNode, RsPingUrmaFindTargetNodeStub, 1);
    ret = RsPongJettyResolveResponsePacket(&pingCb, sgeIdx, &timestamp4);
    EXPECT_INT_EQ(ret, -ENODEV);
    mocker_clean();

    gTmpTarget1.resultSummary.rttMax = 10;
    gTmpTarget1.resultSummary.rttMin = 4;
    pthread_mutex_init(&gTmpTarget1.tripMutex, NULL);
    mocker(RsPingGetTripTime, 1, 11);
    mocker_invoke(RsPingUrmaFindTargetNode, RsPingUrmaFindTargetNodeStub1, 1);
    ret = RsPongJettyResolveResponsePacket(&pingCb, sgeIdx, &timestamp4);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    gTmpTarget1.resultSummary.rttMax = 10;
    gTmpTarget1.resultSummary.rttMin = 4;
    gTmpTarget1.resultSummary.taskAttr.timeoutInterval = 12;
    pthread_mutex_init(&gTmpTarget1.tripMutex, NULL);
    mocker(RsPingGetTripTime, 1, 11);
    mocker_invoke(RsPingUrmaFindTargetNode, RsPingUrmaFindTargetNodeStub1, 1);
    ret = RsPongJettyResolveResponsePacket(&pingCb, sgeIdx, &timestamp4);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    pthread_mutex_destroy(&gTmpTarget1.tripMutex);
    free(pingCb.pongJetty.recvSegCb.sgeList);
    pingCb.pongJetty.recvSegCb.sgeList = NULL;
    free(recvAddr);
    recvAddr = NULL;
}

void TcRsPingCommonImportJetty()
{
    urma_target_jetty_t* importTjetty = NULL;
    struct PingQpInfo target = {0};
    urma_context_t urmaCtx = {0};
    int ret = 0;

    mocker(RsGetJettyInfo, 1, 0);
    ret = RsPingCommonImportJetty(&urmaCtx, &target, &importTjetty);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
    free(importTjetty);
    importTjetty = NULL;
}

void TcRsPingUrmaResetRecvBuffer()
{
    struct RsPingCtxCb pingCb = {0};

    RsPingUrmaResetRecvBuffer(&pingCb);
}

void TcRsPingCommonJfrPostRecv()
{
    struct RsPingLocalJettyCb jettyCb = {0};
    urma_sge_t sge = {0};
    int ret = 0;

    jettyCb.recvSegCb.sgeNum = 1;
    jettyCb.recvSegCb.sgeList = &sge;
    ret = RsPingCommonJfrPostRecv(&jettyCb);
    EXPECT_INT_EQ(ret, 0);
}
