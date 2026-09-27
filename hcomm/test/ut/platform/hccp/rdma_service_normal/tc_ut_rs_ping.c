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
#include "dl_ibverbs_function.h"
#include "rs_socket.h"
#include "ut_dispatch.h"
#include "ra_rs_comm.h"
#include "rs.h"
#include "hccp_common.h"
#include "rs_inner.h"
#include "rs_ping_inner.h"
#include "hccp_ping.h"
#include "rs_ping.h"
#include "rs_ping_roce.h"
#include "tc_ut_rs_ping.h"

extern int RsPingCbGetIbCtxAndIndex(struct rdev* rdevInfo, struct RsPingCtxCb* pingCb);
extern void RsEpollCtl(int epollfd, int op, int fd, int state);
extern int RsGetPingCb(struct RaRsDevInfo* rdev, struct RsPingCtxCb** pingCb);
extern int RsPingCommonPostRecv(struct RsPingLocalQpCb* qpCb);
extern int RsPingCommonInitPostRecvAll(struct RsPingLocalQpCb* qpCb);
extern void RsPingCommonDeinitLocalBuffer(struct RsPingCtxCb* pingCb);
extern int RsPingCbInitMutex(struct RsPingCtxCb* pingCb);
extern void* RsPingHandle(void* arg);
extern int RsPingPongInitLocalBuffer(
    struct rs_cb* rscb, struct PingInitAttr* attr, struct PingInitInfo* info, struct RsPingCtxCb* pingCb);
extern int RsPingCommonInitLocalQp(
    struct rs_cb* rscb, struct RsPingCtxCb* pingCb, union PingQpAttr* attr, struct RsPingLocalQpCb* qpCb);
extern int
RsPingRoceFindTargetNode(struct RsPingCtxCb* pingCb, struct PingQpInfo* target, struct RsPingTargetInfo** node);
extern int RsPingRocePostSend(struct RsPingCtxCb* pingCb, struct RsPingTargetInfo* target);
extern int
RsPingRoceGetTargetResult(struct RsPingCtxCb* pingCb, struct PingTargetCommInfo* target, struct PingResultInfo* result);
extern int
RsPingRoceAllocTargetNode(struct RsPingCtxCb* pingCb, struct PingTargetInfo* target, struct RsPingTargetInfo** node);
extern bool RsPingCommonCompareRdmaInfo(struct PingQpInfo* a, struct PingQpInfo* b);
extern int RsPongFindTargetNode(struct RsPingCtxCb* pingCb, struct PingQpInfo* target, struct RsPongTargetInfo** node);
extern int
RsPongFindAllocTargetNode(struct RsPingCtxCb* pingCb, struct PingQpInfo* target, struct RsPongTargetInfo** node);
extern int RsPingCommonCreateAh(
    struct RsPingCtxCb* pingCb, struct PingLocalCommInfo* localInfo, struct PingQpInfo* remoteInfo, struct ibv_ah** ah);
extern int RsPingCommonPollScq(struct RsPingLocalQpCb* qpCb);
extern int RsPongPostSend(struct RsPingCtxCb* pingCb, struct ibv_wc* wc, struct timeval* timestamp2);
extern int RsPingRocePollRcq(struct RsPingCtxCb* pingCb, int* polledCnt, struct timeval* timestamp2);
extern void RsPongRoceHandleSend(struct RsPingCtxCb* pingCb, int polledCnt, struct timeval* timestamp2);
extern int RsPongResolveResponsePacket(struct RsPingCtxCb* pingCb, uint32_t sgeIdx, struct timeval* timestamp4);
extern void RsPongRocePollRcq(struct RsPingCtxCb* pingCb);
extern int RsPingCbGetDevRdevIndex(struct RsPingCtxCb* pingCb, int index);
extern int RsPingCommonInitMrCb(struct rs_cb* rscb, struct RsPingCtxCb* pingCb, struct RsPingMrCb* mrCb);
extern void RsPingCommonDeinitMrCb(struct RsPingMrCb* mrCb);
extern struct ibv_mr* RsDrvMrReg(struct ibv_pd* pd, char* addr, size_t length, int access);
extern int RsDrvMrDereg(struct ibv_mr* ibMr);
extern int RsPingCommonModifyLocalQp(struct RsPingCtxCb* pingCb, struct RsPingLocalQpCb* qpCb);
extern void RsPingCommonDeinitLocalQp(struct rs_cb* rscb, struct RsPingCtxCb* pingCb, struct RsPingLocalQpCb* qpCb);
extern int RsPingPongInitLocalInfo(
    struct rs_cb* rscb, struct PingInitAttr* attr, struct PingInitInfo* info, struct RsPingCtxCb* pingCb);
extern int RsPingRocePollScq(struct RsPingCtxCb* pingCb, struct RsPingTargetInfo* target);
extern void RsPingPongDelTargetList(struct RsPingCtxCb* pingCb);
extern void RsPingRocePingCbDeinit(unsigned int phyId, struct RsPingCtxCb* pingCb);
extern int RsPingRocePingCbInit(
    unsigned int phyId, struct PingInitAttr* attr, struct PingInitInfo* info, unsigned int* devIndex,
    struct RsPingCtxCb* pingCb);

static struct rs_cb gTmpRsCb0;
static struct rs_cb gTmpRsCb;
static struct rs_cb gTmpRsCb1;
static struct rs_cb gTmpRsCbT;
static struct RsPingCtxCb gTmpPingCb;
static struct ibv_cq gTmpCq;
static struct RsPingTargetInfo gTmpTarget;
static struct RsPingTargetInfo gTmpTarget1;
static struct ibv_mr gIbMr;

int RsDev2rscb_stub0(uint32_t devId, struct rs_cb** rsCb, bool initFlag)
{
    *rsCb = &gTmpRsCb0;

    return 0;
}

int RsGetRsCbStub(unsigned int phyId, struct rs_cb** rsCb)
{
    *rsCb = &gTmpRsCb;
    (*rsCb)->pingCb.pingPongOps = RsPingRoceGetOps();
    (*rsCb)->pingCb.pingPongDfx = RsPingRoceGetDfx();
    return 0;
}

int RsGetRsCbStubTrue(unsigned int phyId, struct rs_cb** rsCb)
{
    *rsCb = &gTmpRsCbT;
    (*rsCb)->pingCb.threadStatus = RS_PING_THREAD_RUNNING;
    return 0;
}

int RsGetRsCbStub1(unsigned int phyId, struct rs_cb** rsCb)
{
    gTmpRsCb1.pingCb.threadStatus = RS_PING_THREAD_RUNNING;

    *rsCb = &gTmpRsCb1;
    (*rsCb)->pingCb.pingPongOps = RsPingRoceGetOps();
    (*rsCb)->pingCb.pingPongDfx = RsPingRoceGetDfx();
    return 0;
}

int RsGetPingCbStub(struct RaRsDevInfo* rdev, struct RsPingCtxCb** pingCb)
{
    *pingCb = &gTmpPingCb;
    (*pingCb)->pingPongOps = RsPingRoceGetOps();
    (*pingCb)->pingPongDfx = RsPingRoceGetDfx();
    return 0;
}

int RsPingRoceFindTargetNodeStub(struct RsPingCtxCb* pingCb, struct PingQpInfo* target, struct RsPingTargetInfo** node)
{
    *node = &gTmpTarget;

    return -ENODEV;
}

int RsPingRoceFindTargetNodeStub1(struct RsPingCtxCb* pingCb, struct PingQpInfo* target, struct RsPingTargetInfo** node)
{
    *node = &gTmpTarget1;

    return 0;
}

int RsPingRoceFindTargetNodeStub2(struct RsPingCtxCb* pingCb, struct PingQpInfo* target, struct RsPingTargetInfo** node)
{
    *node = calloc(1, sizeof(struct RsPingTargetInfo));
    RS_INIT_LIST_HEAD(&(*node)->list);
    (*node)->payloadSize = 1;
    (*node)->payloadBuffer = malloc(1);

    return 0;
}

int RsPongFindTargetNodeStub(struct RsPingCtxCb* pingCb, struct PingQpInfo* target, struct RsPongTargetInfo** node)
{
    *node = NULL;
    return -19;
}

int RsPongFindTargetNodeStub2(struct RsPingCtxCb* pingCb, struct PingQpInfo* target, struct RsPongTargetInfo** node)
{
    *node = calloc(1, sizeof(struct RsPongTargetInfo));
    RS_INIT_LIST_HEAD(&(*node)->list);
    return 0;
}

int RsPongFindAllocTargetNodeStub(struct RsPingCtxCb* pingCb, struct PingQpInfo* target, struct RsPongTargetInfo** node)
{
    static struct RsPongTargetInfo tmpNode = {0};

    *node = &tmpNode;

    return 0;
}

int RsIbvGetCqEvent_stub(struct ibv_comp_channel* channel, struct ibv_cq** cq, void** cqContext)
{
    *cq = &gTmpCq;

    return 0;
}

struct ibv_mr* RsDrvMrRegStub(struct ibv_pd* pd, char* addr, size_t length, int access) { return &gIbMr; }

int RsDrvMrDeregStub(struct ibv_mr* ibMr) { return 0; }

void TcRsPayloadHeaderResvCustomCheck()
{
    /* the llt verifies the size of the RS_PING_PAYLOAD_HEADER_RESV_CUSTOM field */
    EXPECT_INT_EQ(RS_PING_PAYLOAD_HEADER_RESV_CUSTOM, 216);
}

void TcRsPingHandleInit()
{
    unsigned int whiteListStatus = WHITE_LIST_DISABLE;
    int hdcType = HDC_SERVICE_TYPE_RDMA_V2;
    unsigned int chipId = 0;
    int ret = 0;

    mocker_invoke(RsDev2rscb, RsDev2rscb_stub0, 1);
    ret = RsPingHandleInit(hdcType, chipId, whiteListStatus);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsPingHandleDeinit()
{
    unsigned int chipId = 0;
    int ret = 0;

    gTmpRsCb.pingCb.threadStatus = RS_PING_THREAD_RUNNING;
    pthread_mutex_init(&gTmpRsCb.pingCb.pingMutex, NULL);
    mocker_invoke(RsDev2rscb, RsDev2rscb_stub0, 1);
    ret = RsPingHandleDeinit(chipId);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
    pthread_mutex_destroy(&gTmpRsCb.pingCb.pingMutex);
}

void TcRsPingInit()
{
    struct RsPingLocalQpCb qpCb = {0};
    union PingQpAttr rdmaAttr = {0};
    struct RsPingCtxCb pingCb = {0};
    struct ibv_context context = {0};
    struct PingInitAttr attr = {0};
    struct PingInitInfo info = {0};
    struct RaRsDevInfo rdev = {0};
    struct rs_cb tmpRsCb = {0};
    struct ibv_pd tmpPd = {0};
    unsigned int rdevIndex = 0;
    struct ibv_qp ibQp = {0};
    struct ibv_pd pd = {0};
    int ret = 0;

    mocker_invoke(RsGetRsCb, RsGetRsCbStub, 20);
    mocker(rsGetLocalDevIDByHostDevID, 20, 0);
    mocker(RsInetNtop, 20, 0);
    mocker(RsSetupSharemem, 20, 0);
    ret = RsPingInit(&attr, &info, &rdevIndex);
    EXPECT_INT_EQ(ret, -ENODEV);
    mocker_clean();

    mocker_invoke(RsGetRsCb, RsGetRsCbStubTrue, 20);
    mocker(rsGetLocalDevIDByHostDevID, 20, 0);
    mocker(RsInetNtop, 20, 0);
    mocker(RsSetupSharemem, 20, 0);
    mocker(RsPingRocePingCbInit, 20, 0);
    ret = RsPingInit(&attr, &info, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    qpCb.ibQp = &ibQp;
    mocker(RsIbvQueryQp, 20, -1);
    ret = RsPingCommonModifyLocalQp(&pingCb, &qpCb);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker(RsIbvQueryQp, 20, 0);
    mocker(RsIbvModifyQp, 20, -1);
    ret = RsPingCommonModifyLocalQp(&pingCb, &qpCb);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker(RsIbvQueryQp, 20, 0);
    mocker(RsIbvModifyQp, 20, 0);
    ret = RsPingCommonModifyLocalQp(&pingCb, &qpCb);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)RsIbvCreateCq, 10, NULL);
    ret = RsPingCommonInitLocalQp(&tmpRsCb, &pingCb, &rdmaAttr, &qpCb);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    pingCb.rdevCb.ibCtx = &context;
    pingCb.rdevCb.ibPd = &tmpPd;
    mocker(RsEpollCtl, 20, 0);
    mocker(RsIbvExpCreateQp, 10, 0);
    ret = RsPingCommonInitLocalQp(&tmpRsCb, &pingCb, &rdmaAttr, &qpCb);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker(RsPingCommonInitMrCb, 20, -1);
    ret = RsPingPongInitLocalBuffer(&tmpRsCb, &attr, &info, &pingCb);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker(RsPingCommonInitMrCb, 20, 0);
    ret = RsPingPongInitLocalBuffer(&tmpRsCb, &attr, &info, &pingCb);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    qpCb.recvMrCb.sgeNum = 1;
    qpCb.qpCap.max_recv_wr = 1;
    mocker(RsPingCommonPostRecv, 20, 0);
    ret = RsPingCommonInitPostRecvAll(&qpCb);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    qpCb.recvMrCb.sgeNum = 1;
    qpCb.qpCap.max_recv_wr = 1;
    mocker(RsPingCommonPostRecv, 20, -1);
    ret = RsPingCommonInitPostRecvAll(&qpCb);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRsPingTargetAdd()
{
    struct PingTargetInfo target = {0};
    struct RaRsDevInfo rdev = {0};
    int ret = 0;

    target.localInfo.rdma.hopLimit = 64;
    target.localInfo.rdma.qosAttr.tc = (33 & 0x3f) << 2;
    target.localInfo.rdma.qosAttr.sl = 4;
    gTmpPingCb.taskStatus = RS_PING_TASK_RESET;
    RS_INIT_LIST_HEAD(&gTmpPingCb.pingList);
    mocker_invoke(RsGetPingCb, RsGetPingCbStub, 1);
    ret = RsPingTargetAdd(&rdev, &target);
    EXPECT_INT_EQ(ret, -9);
    mocker_clean();

    target.payload.size = 1;
    gTmpPingCb.taskStatus = RS_PING_TASK_RESET;
    RS_INIT_LIST_HEAD(&gTmpPingCb.pingList);
    mocker_invoke(RsGetPingCb, RsGetPingCbStub, 1);
    mocker(RsPingCommonCreateAh, 1, -1);
    ret = RsPingTargetAdd(&rdev, &target);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker_invoke(RsGetPingCb, RsGetPingCbStub, 1);
    mocker(calloc, 10, 0);
    ret = RsPingTargetAdd(&rdev, &target);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();
}

void TcRsGetPingCb()
{
    struct RaRsDevInfo rdev = {0};
    struct RsPingCtxCb* pingCb = NULL;
    int ret = 0;

    mocker(RsGetRsCb, 1, -1);
    ret = RsGetPingCb(&rdev, &pingCb);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker_invoke(RsGetRsCb, RsGetRsCbStub1, 1);
    ret = RsGetPingCb(&rdev, &pingCb);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsPingClientPostSend()
{
    struct RsPingTargetInfo target = {0};
    struct RsPingCtxCb pingCb = {0};
    struct ibv_qp serverIbQp = {0};
    struct ibv_qp clientIbQp = {0};
    struct ibv_sge sge = {0};
    void* addr = malloc(256);
    int ret = 0;

    target.payloadBuffer = malloc(1);
    target.payloadSize = 1;
    sge.addr = (uintptr_t)addr;
    sge.length = 256;
    pingCb.pingQp.sendMrCb.sgeNum = 1;
    pingCb.pingQp.sendMrCb.sgeList = &sge;
    pingCb.pongQp.ibQp = &serverIbQp;
    pingCb.pingQp.ibQp = &clientIbQp;
    mocker(RsIbvPostSend, 1, -1);
    ret = RsPingRocePostSend(&pingCb, &target);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(RsIbvPostSend, 1, 0);
    ret = RsPingRocePostSend(&pingCb, &target);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    free(addr);
    addr = NULL;
    free(target.payloadBuffer);
    target.payloadBuffer = NULL;
}

void TcRsPingGetResults()
{
    struct PingTargetCommInfo target = {0};
    struct PingResultInfo result = {0};
    struct RaRsDevInfo rdev = {0};
    unsigned int num = 1;
    int ret = 0;

    gTmpPingCb.taskStatus = RS_PING_TASK_RESET;
    mocker_invoke(RsGetPingCb, RsGetPingCbStub, 1);
    mocker_invoke(RsPingRoceFindTargetNode, RsPingRoceFindTargetNodeStub, 1);
    ret = RsPingGetResults(&rdev, &target, &num, &result);
    EXPECT_INT_EQ(ret, -ENODEV);
    mocker_clean();

    num = 1;
    gTmpPingCb.taskStatus = RS_PING_TASK_RESET;
    mocker_invoke(RsGetPingCb, RsGetPingCbStub, 1);
    mocker_invoke(RsPingRoceFindTargetNode, RsPingRoceFindTargetNodeStub1, 1);
    ret = RsPingGetResults(&rdev, &target, &num, &result);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsPingTaskStop()
{
    struct RaRsDevInfo rdev = {0};
    int ret = 0;

    pthread_mutex_init(&gTmpPingCb.pingMutex, NULL);
    mocker_invoke(RsGetPingCb, RsGetPingCbStub, 1);
    ret = RsPingTaskStop(&rdev);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
    pthread_mutex_destroy(&gTmpPingCb.pingMutex);
}

void TcRsPingTargetDel()
{
    struct PingTargetCommInfo target = {0};
    struct RaRsDevInfo rdev = {0};
    unsigned int num = 1;
    int ret = 0;

    gTmpPingCb.taskStatus = RS_PING_TASK_RESET;
    mocker_invoke(RsGetPingCb, RsGetPingCbStub, 1);
    mocker_invoke(RsPingRoceFindTargetNode, RsPingRoceFindTargetNodeStub2, 1);
    ret = RsPingTargetDel(&rdev, &target, &num);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsPingDeinit()
{
    struct ibv_comp_channel clientChannel = {0};
    struct ibv_comp_channel serverChannel = {0};
    struct PingTargetInfo target = {0};
    struct RaRsDevInfo rdev = {0};
    struct rs_cb rsCb = {0};
    int ret = 0;

    gTmpPingCb.initCnt = 1;
    gTmpPingCb.pingQp.channel = &clientChannel;
    gTmpPingCb.pongQp.channel = &serverChannel;
    RS_INIT_LIST_HEAD(&gTmpPingCb.pingList);
    RS_INIT_LIST_HEAD(&gTmpPingCb.pingList);
    target.payload.size = 1;
    mocker_invoke(RsGetPingCb, RsGetPingCbStub, 10);
    mocker(RsPingCommonCreateAh, 1, 0);
    ret = RsPingTargetAdd(&rdev, &target);
    EXPECT_INT_EQ(ret, 0);
    mocker(RsEpollCtl, 20, 0);
    mocker(RsIbvDestroyCompChannel, 20, 0);
    ret = RsPingDeinit(&rdev);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsPingCompareRdmaInfo()
{
    struct PingQpInfo a = {0};
    struct PingQpInfo b = {0};
    int ret = 0;

    ret = RsPingCommonCompareRdmaInfo(&a, &b);
    EXPECT_INT_EQ(ret, true);

    a.rdma.qpn = 1;
    b.rdma.qpn = 2;
    ret = RsPingCommonCompareRdmaInfo(&a, &b);
    EXPECT_INT_EQ(ret, false);

    a.rdma.qpn = 1;
    b.rdma.qpn = 1;
    a.rdma.qkey = 1;
    b.rdma.qkey = 2;
    ret = RsPingCommonCompareRdmaInfo(&a, &b);
    EXPECT_INT_EQ(ret, false);

    a.rdma.qpn = 1;
    b.rdma.qpn = 1;
    a.rdma.qkey = 1;
    b.rdma.qkey = 1;
    a.rdma.gid.raw[0] = 1;
    b.rdma.gid.raw[0] = 2;
    ret = RsPingCommonCompareRdmaInfo(&a, &b);
    EXPECT_INT_EQ(ret, false);
}

void TcRsPingRoceFindTargetNode()
{
    struct RsPingTargetInfo stubNode = {0};
    struct RsPingTargetInfo* node = NULL;
    struct RsPingCtxCb pingCb = {0};
    struct PingQpInfo target = {0};
    int ret = 0;

    RS_INIT_LIST_HEAD(&pingCb.pingList);
    ret = RsPingRoceFindTargetNode(&pingCb, &target, &node);
    EXPECT_INT_EQ(ret, -ENODEV);

    RsListAddTail(&stubNode.list, &pingCb.pingList);
    ret = RsPingRoceFindTargetNode(&pingCb, &target, &node);
    EXPECT_INT_EQ(ret, 0);
}

void TcRsPongFindTargetNode()
{
    struct RsPongTargetInfo stubNode = {0};
    struct RsPongTargetInfo* node = NULL;
    struct RsPingCtxCb pingCb = {0};
    struct PingQpInfo target = {0};
    int ret = 0;

    RS_INIT_LIST_HEAD(&pingCb.pongList);
    ret = RsPongFindTargetNode(&pingCb, &target, &node);
    EXPECT_INT_EQ(ret, -ENODEV);

    RsListAddTail(&stubNode.list, &pingCb.pongList);
    ret = RsPongFindTargetNode(&pingCb, &target, &node);
    EXPECT_INT_EQ(ret, 0);
}

void TcRsPongFindAllocTargetNode()
{
    struct RsPongTargetInfo* node = NULL;
    struct RsPingCtxCb pingCb = {0};
    struct PingQpInfo target = {0};
    int ret = 0;

    mocker_invoke(RsPongFindTargetNode, RsPongFindTargetNodeStub2, 1);
    mocker(RsPingCommonCreateAh, 1, -1);
    ret = RsPongFindAllocTargetNode(&pingCb, &target, &node);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    pthread_mutex_init(&pingCb.pongMutex, NULL);
    RS_INIT_LIST_HEAD(&pingCb.pongList);
    mocker_invoke(RsPongFindTargetNode, RsPongFindTargetNodeStub, 1);
    mocker(RsPingCommonCreateAh, 1, 0);
    ret = RsPongFindAllocTargetNode(&pingCb, &target, &node);
    free(node);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsPingPollSendCq()
{
    struct RsPingLocalQpCb qpCb = {0};
    int ret = 0;

    mocker(RsIbvPollCq, 1, -1);
    ret = RsPingCommonPollScq(&qpCb);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker(RsIbvPollCq, 1, 1);
    ret = RsPingCommonPollScq(&qpCb);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsPingServerPostSend()
{
    struct RsPingCtxCb pingCb = {0};
    struct timeval timestamp2 = {0};
    void* sendAddr = malloc(1024);
    void* recvAddr = malloc(1024);
    struct ibv_wc wc = {0};
    int ret = 0;

    wc.wr_id = 1;
    mocker(RsPingCommonPollScq, 1, 0);
    ret = RsPongPostSend(&pingCb, &wc, &timestamp2);
    EXPECT_INT_EQ(ret, -EIO);
    mocker_clean();

    wc.wr_id = 0;
    wc.byte_len = 16 + RS_PING_PAYLOAD_HEADER_RESV_GRH;
    pingCb.pingQp.recvMrCb.sgeList = calloc(1, sizeof(struct ibv_sge));
    pingCb.pingQp.recvMrCb.sgeNum = 1;
    pingCb.pongQp.sendMrCb.sgeList = calloc(1, sizeof(struct ibv_sge));
    pingCb.pongQp.sendMrCb.sgeNum = 1;
    pingCb.pongQp.sendMrCb.sgeList->addr = (uintptr_t)sendAddr;
    pingCb.pongQp.sendMrCb.sgeList->length = 1024;
    pingCb.pingQp.recvMrCb.sgeList->addr = (uintptr_t)recvAddr;
    pingCb.pingQp.recvMrCb.sgeList->length = 1024;
    mocker(RsPingCommonPollScq, 1, 0);
    mocker(RsPongFindAllocTargetNode, 1, -1);
    ret = RsPongPostSend(&pingCb, &wc, &timestamp2);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(RsPingCommonPollScq, 1, 0);
    mocker_invoke(RsPongFindAllocTargetNode, RsPongFindAllocTargetNodeStub, 1);
    mocker(gettimeofday, 20, 1);
    mocker(RsIbvPostSend, 1, -1);
    ret = RsPongPostSend(&pingCb, &wc, &timestamp2);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(RsPingCommonPollScq, 1, 0);
    mocker_invoke(RsPongFindAllocTargetNode, RsPongFindAllocTargetNodeStub, 1);
    mocker(RsIbvPostSend, 1, 0);
    ret = RsPongPostSend(&pingCb, &wc, &timestamp2);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    free(pingCb.pongQp.sendMrCb.sgeList);
    pingCb.pongQp.sendMrCb.sgeList = NULL;
    free(pingCb.pingQp.recvMrCb.sgeList);
    pingCb.pingQp.recvMrCb.sgeList = NULL;
    free(recvAddr);
    recvAddr = NULL;
    free(sendAddr);
    sendAddr = NULL;
}

void TcRsPingPostRecv()
{
    struct RsPingLocalQpCb qpCb = {0};
    void* recvAddr = malloc(1024);
    int ret = 0;

    qpCb.recvMrCb.sgeList = calloc(1, sizeof(struct ibv_sge));
    qpCb.recvMrCb.sgeList->addr = (uintptr_t)recvAddr;
    qpCb.recvMrCb.sgeList->length = 1024;
    qpCb.recvMrCb.sgeNum = 1;
    mocker(RsIbvPostRecv, 1, -1);
    ret = RsPingCommonPostRecv(&qpCb);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(RsIbvPostRecv, 1, 0);
    ret = RsPingCommonPostRecv(&qpCb);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    free(qpCb.recvMrCb.sgeList);
    qpCb.recvMrCb.sgeList = NULL;
    free(recvAddr);
    recvAddr = NULL;
}

void TcRsPingClientPollCq()
{
    struct RsPingCtxCb pingCb = {0};
    struct timeval timestamp2 = {0};
    int polledCnt = 0;
    int ret = 0;

    mocker_clean();
    mocker(RsIbvGetCqEvent, 1, -1);
    ret = RsPingRocePollRcq(&pingCb, &polledCnt, &timestamp2);
    EXPECT_INT_EQ(ret, -259);
    mocker_clean();

    mocker(RsIbvGetCqEvent, 1, 0);
    mocker(RsIbvPollCq, 1, 0);
    ret = RsPingRocePollRcq(&pingCb, &polledCnt, &timestamp2);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    pingCb.pingQp.recvCq.ibCq = &gTmpCq;
    mocker_invoke(RsIbvGetCqEvent, RsIbvGetCqEvent_stub, 1);
    mocker(RsIbvPollCq, 1, -1);
    ret = RsPingRocePollRcq(&pingCb, &polledCnt, &timestamp2);
    EXPECT_INT_EQ(ret, -259);
    mocker_clean();

    pingCb.pingQp.recvCq.maxRecvWcNum = 16;
    mocker_invoke(RsIbvGetCqEvent, RsIbvGetCqEvent_stub, 1);
    mocker(RsIbvPollCq, 1, 1);
    mocker(RsPongPostSend, 1, -1);
    ret = RsPingRocePollRcq(&pingCb, &polledCnt, &timestamp2);
    EXPECT_INT_EQ(ret, 0);
    RsPongRoceHandleSend(&pingCb, polledCnt, &timestamp2);
    mocker_clean();

    mocker_invoke(RsIbvGetCqEvent, RsIbvGetCqEvent_stub, 1);
    mocker(RsIbvPollCq, 1, 1);
    mocker(RsPongPostSend, 1, 0);
    mocker(RsPingCommonPostRecv, 1, -1);
    ret = RsPingRocePollRcq(&pingCb, &polledCnt, &timestamp2);
    EXPECT_INT_EQ(ret, 0);
    RsPongRoceHandleSend(&pingCb, polledCnt, &timestamp2);
    mocker_clean();

    mocker_invoke(RsIbvGetCqEvent, RsIbvGetCqEvent_stub, 1);
    mocker(RsIbvPollCq, 1, 1);
    mocker(RsPongPostSend, 1, 0);
    mocker(RsPingCommonPostRecv, 1, 0);
    mocker(RsIbvReqNotifyCq, 1, -1);
    ret = RsPingRocePollRcq(&pingCb, &polledCnt, &timestamp2);
    EXPECT_INT_EQ(ret, 0);
    RsPongRoceHandleSend(&pingCb, polledCnt, &timestamp2);
    mocker_clean();
}

void TcRsEpollEventPingHandle()
{
    struct ibv_comp_channel pingChannel = {0};
    struct ibv_comp_channel pongChannel = {0};
    struct rs_cb rscb = {0};
    int ret = 0;

    pongChannel.fd = 1;
    pthread_mutex_init(&rscb.pingCb.devMutex, NULL);
    rscb.pingCb.initCnt = 1;
    rscb.pingCb.pingQp.channel = &pingChannel;
    rscb.pingCb.pongQp.channel = &pongChannel;
    rscb.pingCb.threadStatus = RS_PING_THREAD_RUNNING;
    rscb.pingCb.pingPongOps = RsPingRoceGetOps();
    rscb.pingCb.pingPongDfx = RsPingRoceGetDfx();

    mocker(RsPingRocePollRcq, 10, 0);
    mocker(RsPongRoceHandleSend, 10, 0);
    mocker(RsPongRocePollRcq, 10, 0);

    ret = RsEpollEventPingHandle(&rscb, 0);
    EXPECT_INT_EQ(ret, 0);
    ret = RsEpollEventPingHandle(&rscb, 1);
    EXPECT_INT_EQ(ret, 0);
    rscb.pingCb.initCnt = 0;
    ret = RsEpollEventPingHandle(&rscb, 1);
    EXPECT_INT_EQ(ret, -ENODEV);
    mocker_clean();
    return;
}

void TcRsPingGetTripTime()
{
    struct RsPingTimestamp timestamp = {0};
    unsigned int ret = 0;

    ret = RsPingGetTripTime(&timestamp);
    EXPECT_INT_EQ(ret, 0);
}

int PthreadMutexInitStub2(pthread_mutex_t lock, void* ptr)
{
    static int cnt = 0;
    cnt++;
    if (cnt == 2) {
        return -5;
    }
    return 0;
}

int PthreadMutexInitStub3(pthread_mutex_t lock, void* ptr)
{
    static int cnt = 0;
    cnt++;
    if (cnt == 3) {
        return -5;
    }
    return 0;
}

void TcRsPingCbInitMutexAb1()
{
    struct RsPingCtxCb pingCb = {0};
    int ret = 0;
    mocker(pthread_mutex_init, 10, -5);
    ret = RsPingCbInitMutex(&pingCb);
    EXPECT_INT_EQ(ret, -258);
    mocker_clean();
}

void TcRsPingCbInitMutexAb2()
{
    struct RsPingCtxCb pingCb = {0};
    int ret = 0;
    mocker_invoke(pthread_mutex_init, PthreadMutexInitStub2, 10);
    ret = RsPingCbInitMutex(&pingCb);
    EXPECT_INT_EQ(ret, -258);
    mocker_clean();
}

void TcRsPingCbInitMutexAb3()
{
    struct RsPingCtxCb pingCb = {0};
    int ret = 0;
    mocker_invoke(pthread_mutex_init, PthreadMutexInitStub3, 10);
    ret = RsPingCbInitMutex(&pingCb);
    EXPECT_INT_EQ(ret, -258);
    mocker_clean();
}

void TcRsPingCbInitMutex()
{
    TcRsPingCbInitMutexAb1();
    TcRsPingCbInitMutexAb2();
    TcRsPingCbInitMutexAb3();
}

void TcRsPingResolveResponsePacket()
{
    struct RsPingCtxCb pingCb = {0};
    struct timeval timestamp4 = {0};
    void* recvAddr = calloc(1, 1024);
    uint32_t sgeIdx = 0;
    int ret = 0;

    pingCb.pongQp.recvMrCb.sgeList = calloc(1, sizeof(struct ibv_sge));
    pingCb.pongQp.recvMrCb.sgeList->addr = (uintptr_t)recvAddr;
    pingCb.pongQp.recvMrCb.sgeList->length = 1024;
    pingCb.taskId = 1;

    ret = RsPongResolveResponsePacket(&pingCb, sgeIdx, &timestamp4);
    EXPECT_INT_EQ(ret, 0);

    pingCb.taskId = 0;
    mocker_invoke(RsPingRoceFindTargetNode, RsPingRoceFindTargetNodeStub, 1);
    ret = RsPongResolveResponsePacket(&pingCb, sgeIdx, &timestamp4);
    EXPECT_INT_EQ(ret, -ENODEV);
    mocker_clean();

    gTmpTarget1.resultSummary.rttMax = 10;
    gTmpTarget1.resultSummary.rttMin = 4;
    pthread_mutex_init(&gTmpTarget1.tripMutex, NULL);
    mocker(RsPingGetTripTime, 1, 11);
    mocker_invoke(RsPingRoceFindTargetNode, RsPingRoceFindTargetNodeStub1, 1);
    ret = RsPongResolveResponsePacket(&pingCb, sgeIdx, &timestamp4);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    gTmpTarget1.resultSummary.rttMax = 10;
    gTmpTarget1.resultSummary.rttMin = 4;
    gTmpTarget1.resultSummary.taskAttr.timeoutInterval = 12;
    pthread_mutex_init(&gTmpTarget1.tripMutex, NULL);
    mocker(RsPingGetTripTime, 1, 11);
    mocker_invoke(RsPingRoceFindTargetNode, RsPingRoceFindTargetNodeStub1, 1);
    ret = RsPongResolveResponsePacket(&pingCb, sgeIdx, &timestamp4);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    pthread_mutex_destroy(&gTmpTarget1.tripMutex);
    free(pingCb.pongQp.recvMrCb.sgeList);
    pingCb.pongQp.recvMrCb.sgeList = NULL;
    free(recvAddr);
    recvAddr = NULL;
}

void TcRsPingServerPollCq()
{
    struct RsPingCtxCb pingCb = {0};

    mocker(RsIbvGetCqEvent, 1, -1);
    RsPongRocePollRcq(&pingCb);
    mocker_clean();

    mocker(RsIbvGetCqEvent, 1, 0);
    RsPongRocePollRcq(&pingCb);
    mocker_clean();

    pingCb.pongQp.recvCq.ibCq = &gTmpCq;
    pingCb.pongQp.recvCq.maxRecvWcNum = 16;
    mocker_invoke(RsIbvGetCqEvent, RsIbvGetCqEvent_stub, 1);
    mocker(RsIbvPollCq, 1, -1);
    RsPongRocePollRcq(&pingCb);
    mocker_clean();

    mocker_invoke(RsIbvGetCqEvent, RsIbvGetCqEvent_stub, 1);
    mocker(RsIbvPollCq, 1, 1);
    mocker(RsPongResolveResponsePacket, 1, -1);
    RsPongRocePollRcq(&pingCb);
    mocker_clean();

    mocker_invoke(RsIbvGetCqEvent, RsIbvGetCqEvent_stub, 1);
    mocker(RsIbvPollCq, 1, 1);
    mocker(RsPongResolveResponsePacket, 1, 0);
    mocker(RsPingCommonPostRecv, 1, -1);
    RsPongRocePollRcq(&pingCb);
    mocker_clean();

    mocker_invoke(RsIbvGetCqEvent, RsIbvGetCqEvent_stub, 1);
    mocker(RsIbvPollCq, 1, 1);
    mocker(RsPongResolveResponsePacket, 1, 0);
    mocker(RsPingCommonPostRecv, 1, 0);
    mocker(RsIbvReqNotifyCq, 1, -1);
    RsPongRocePollRcq(&pingCb);
    mocker_clean();
}

void TcRsPingCbGetDevRdevIndex()
{
#ifndef CUSTOM_INTERFACE
#define CUSTOM_INTERFACE
#endif

    struct ibv_device* devList = calloc(1, sizeof(struct ibv_device));
    struct RsPingCtxCb pingCb = {0};
    int index = 0;
    int ret = 0;

    pthread_mutex_init(&pingCb.pingMutex, NULL);
    pingCb.rdevCb.devList = &devList;
    mocker(RsIbvGetDeviceName, 1, 0);
    mocker(RsRoceGetRoceDevData, 1, -1);
    ret = RsPingCbGetDevRdevIndex(&pingCb, index);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(RsIbvGetDeviceName, 1, 0);
    mocker(RsRoceGetRoceDevData, 1, 0);
    ret = RsPingCbGetDevRdevIndex(&pingCb, index);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    free(devList);
    devList = NULL;
    pthread_mutex_destroy(&pingCb.pingMutex);
}

void TcRsPingInitMrCb()
{
    struct RsPingCtxCb pingCb = {0};
    struct RsPingMrCb mrCb = {0};
    struct rs_cb rscb = {0};
    struct ibv_mr mr = {0};
    int ret = 0;

    mocker((stub_fn_t)pthread_mutex_init, 1, -1);
    ret = RsPingCommonInitMrCb(&rscb, &pingCb, &mrCb);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    ret = RsPingCommonInitMrCb(&rscb, &pingCb, &mrCb);
    EXPECT_INT_NE(ret, 0);

    mocker(DlHalBuffAllocAlignEx, 1, 0);
    mocker(RsDrvMrReg, 1, NULL);
    ret = RsPingCommonInitMrCb(&rscb, &pingCb, &mrCb);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker(DlHalBuffAllocAlignEx, 1, 0);
    mocker_invoke(RsDrvMrReg, RsDrvMrRegStub, 1);
    mocker_invoke(RsDrvMrDereg, RsDrvMrDeregStub, 1);
    mocker(calloc, 10, NULL);
    ret = RsPingCommonInitMrCb(&rscb, &pingCb, &mrCb);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    mocker(DlHalBuffAllocAlignEx, 1, 0);
    mocker_invoke(RsDrvMrReg, RsDrvMrRegStub, 1);
    mocker_invoke(RsDrvMrDereg, RsDrvMrDeregStub, 1);
    mrCb.sgeNum = 1;
    ret = RsPingCommonInitMrCb(&rscb, &pingCb, &mrCb);
    EXPECT_INT_EQ(ret, 0);

    mocker(DlHalBuffFree, 1, 0);
    RsPingCommonDeinitMrCb(&mrCb);
    mocker_clean();
}

void TcRsPingCommonDeinitLocalBuffer()
{
    struct RsPingCtxCb pingCb = {0};
    pingCb.pongQp.recvMrCb.sgeList = calloc(1, sizeof(struct ibv_sge));
    pingCb.pongQp.sendMrCb.sgeList = calloc(1, sizeof(struct ibv_sge));
    pingCb.pingQp.recvMrCb.sgeList = calloc(1, sizeof(struct ibv_sge));
    pingCb.pingQp.sendMrCb.sgeList = calloc(1, sizeof(struct ibv_sge));

    mocker(RsDrvMrDereg, 20, 0);
    mocker(DlHalBuffFree, 20, 0);
    mocker(pthread_mutex_destroy, 20, 0);

    RsPingCommonDeinitLocalBuffer(&pingCb);

    mocker_clean();
}

void TcRsPingCommonDeinitLocalQp()
{
    struct RsPingLocalQpCb qpCb = {0};
    struct ibv_comp_channel channel = {0};
    struct RsPingCtxCb pingCb = {0};
    struct rs_cb rscb = {0};

    qpCb.channel = &channel;
    mocker(RsIbvDestroyQp, 20, 0);
    mocker(pthread_mutex_destroy, 20, 0);
    mocker(RsIbvAckCqEvents, 20, 0);
    mocker(RsIbvDestroyCq, 20, 0);
    mocker(RsEpollCtl, 20, 0);
    mocker(RsIbvDestroyCompChannel, 20, 0);

    RsPingCommonDeinitLocalQp(NULL, NULL, NULL);
    RsPingCommonDeinitLocalQp(&rscb, &pingCb, &qpCb);
    mocker_clean();
}

void TcRsPingPongInitLocalInfo()
{
    struct RsPingCtxCb pingCb = {0};
    struct PingInitAttr attr = {0};
    struct PingInitInfo info = {0};
    struct ibv_qp ibQp = {0};
    struct rs_cb rscb = {0};
    int ret = 0;

    pingCb.pingQp.ibQp = &ibQp;
    pingCb.pongQp.ibQp = &ibQp;

    mocker(RsPingCommonInitLocalQp, 20, 0);
    mocker(RsPingPongInitLocalBuffer, 20, 0);
    mocker(RsPingCommonInitPostRecvAll, 20, 0);
    mocker(RsPingCommonDeinitLocalBuffer, 20, 0);
    mocker(RsPingCommonDeinitLocalQp, 20, 0);
    ret = RsPingPongInitLocalInfo(&rscb, &attr, &info, &pingCb);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker(RsPingCommonInitLocalQp, 20, -1);
    mocker(RsPingPongInitLocalBuffer, 20, 0);
    mocker(RsPingCommonInitPostRecvAll, 20, 0);
    mocker(RsPingCommonDeinitLocalBuffer, 20, 0);
    mocker(RsPingCommonDeinitLocalQp, 20, 0);
    ret = RsPingPongInitLocalInfo(&rscb, &attr, &info, &pingCb);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(RsPingCommonInitLocalQp, 20, 0);
    mocker(RsPingPongInitLocalBuffer, 20, -1);
    mocker(RsPingCommonInitPostRecvAll, 20, 0);
    mocker(RsPingCommonDeinitLocalBuffer, 20, 0);
    mocker(RsPingCommonDeinitLocalQp, 20, 0);
    ret = RsPingPongInitLocalInfo(&rscb, &attr, &info, &pingCb);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(RsPingCommonInitLocalQp, 20, 0);
    mocker(RsPingPongInitLocalBuffer, 20, 0);
    mocker(RsPingCommonInitPostRecvAll, 20, -1);
    mocker(RsPingCommonDeinitLocalBuffer, 20, 0);
    mocker(RsPingCommonDeinitLocalQp, 20, 0);
    ret = RsPingPongInitLocalInfo(&rscb, &attr, &info, &pingCb);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

int RsIbvPollCq_stub_ping0(struct ibv_cq* cq, int numEntries, struct ibv_wc* wc) { return 0; }

int RsIbvPollCq_stub_ping1(struct ibv_cq* cq, int numEntries, struct ibv_wc* wc)
{
    wc->status = IBV_WC_LOC_LEN_ERR;
    return 1;
}

int RsIbvPollCq_stub_ping2(struct ibv_cq* cq, int numEntries, struct ibv_wc* wc)
{
    wc->status = IBV_WC_SUCCESS;
    return 1;
}

void TcRsPingRocePollScq()
{
    struct RsPingTargetInfo target = {0};
    struct RsPingCtxCb pingCb = {0};
    int ret = 0;

    mocker_invoke(RsIbvPollCq, RsIbvPollCq_stub_ping0, 10);
    ret = RsPingRocePollScq(&pingCb, &target);
    EXPECT_INT_EQ(ret, -61);
    mocker_clean();

    mocker_invoke(RsIbvPollCq, RsIbvPollCq_stub_ping1, 10);
    ret = RsPingRocePollScq(&pingCb, &target);
    EXPECT_INT_EQ(ret, -259);
    mocker_clean();

    mocker_invoke(RsIbvPollCq, RsIbvPollCq_stub_ping2, 10);
    ret = RsPingRocePollScq(&pingCb, &target);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

int StubRsPingClientPostSend(struct RsPingCtxCb* pingCb, struct RsPingTargetInfo* target)
{
    pingCb->threadStatus = 0;
    return 1;
}

void TcRsPingHandle()
{
    struct rs_cb rsCb = {0};
    struct RsPingTargetInfo targetTmp = {0};

    mocker_clean();
    rsCb.pingCb.threadStatus = 1;
    rsCb.pingCb.taskStatus = 1;
    rsCb.pingCb.taskAttr.packetCnt = 1;
    RS_INIT_LIST_HEAD(&rsCb.pingCb.pingList);
    rsCb.pingCb.pingList.next = &targetTmp.list;
    rsCb.pingCb.pingList.prev = &targetTmp.list;
    rsCb.pingCb.pingPongOps = RsPingRoceGetOps();
    rsCb.pingCb.pingPongDfx = RsPingRoceGetDfx();
    targetTmp.list.next = &rsCb.pingCb.pingList;
    targetTmp.list.prev = &rsCb.pingCb.pingList;
    targetTmp.state = 1;
    mocker(RsGetCurTime, 1, 0);
    mocker(strncpy_s, 1, 0);
    mocker(RsHeartbeatAlivePrint, 1, 0);
    mocker(RsListEmpty, 1, 0);
    mocker_invoke(RsPingRocePostSend, StubRsPingClientPostSend, 1);
    mocker(pthread_mutex_lock, 1, 0);
    mocker(pthread_mutex_unlock, 1, 0);
    RsPingHandle((void*)&rsCb);
    mocker_clean();
}

void TcRsPingRocePingCbDeinit()
{
    struct RsPingTargetInfo* stubPingNode = NULL;
    struct RsPongTargetInfo* stubPongNode = NULL;
    struct RsPingCtxCb pingCb = {0};

    mocker(RsGetRsCb, 20, 0);
    mocker(pthread_mutex_lock, 20, 0);
    mocker(pthread_mutex_unlock, 20, 0);

    stubPingNode = calloc(1, sizeof(struct RsPingTargetInfo));
    stubPongNode = calloc(1, sizeof(struct RsPongTargetInfo));
    RS_INIT_LIST_HEAD(&pingCb.pingList);
    RsListAddTail(&stubPingNode->list, &pingCb.pingList);
    RS_INIT_LIST_HEAD(&pingCb.pongList);
    RsListAddTail(&stubPongNode->list, &pingCb.pongList);
    mocker(RsIbvDestroyAh, 20, 0);

    mocker(RsPingCommonDeinitLocalQp, 20, 0);
    mocker(RsPingCommonDeinitLocalBuffer, 20, 0);
    mocker(RsIbvDeallocPd, 20, 0);
    mocker(RsIbvCloseDevice, 20, 0);
    mocker(RsIbvFreeDeviceList, 20, 0);

    RsPingRocePingCbDeinit(0, &pingCb);
}
